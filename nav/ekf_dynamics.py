"""
Shared EKF dynamics model for Earth-centred orbital navigation.
9-state: [r (3 km) | v (3 km/s) | clkb (km) | clkdr (km/s) | gamma_srp (m^2/kg)]
"""
import numpy as np
import AA278.project.orbital.ephemeris as ephemeris
from AA278.project.misc.constants import (
    MU_EARTH, MU_MOON, MU_SUN, R_EARTH, J2, P_SR, AU_KM, C_LIGHT,
)

# SRP scale constant
SRP_COEFF = (P_SR / (C_LIGHT * 1e3)) * AU_KM**2 * 1e-3

# RAFS-class clock noise PSDs
CLK_Q1 = 3.70e-24 * C_LIGHT**2
CLK_Q2 = 1.87e-33 * C_LIGHT**2


def j2_accel_and_jacobian(r_vec):
    """
    J2 oblateness acceleration [km/s^2] and its 3x3 position Jacobian.

    Derivation: a_J2 = K/r^5 * v  where v = [x(c-1), y(c-1), z(c-3)], c = 5z^2/r^2
    Jacobian:   G_J2 = K/r^5 * dv_dr - 5K/r^7 * outer(v, r)
    """
    x, y, z = r_vec
    r2 = x*x + y*y + z*z
    r  = np.sqrt(r2)
    r4 = r2 * r2
    r5 = r4 * r
    r7 = r4 * r2 * r

    K = 1.5 * J2 * MU_EARTH * R_EARTH ** 2
    c = 5.0 * z * z / r2

    v = np.array([x * (c - 1.0), y * (c - 1.0), z * (c - 3.0)])
    a = (K / r5) * v

    # d(c)/d(r): shape (3,)
    dc_dr = np.array([-10.0*x*z*z / r4,
                      -10.0*y*z*z / r4,
                       10.0*z*(r2 - z*z) / r4])

    # d(v)/d(r): off-diagonal = r[i]*dc_dr[j]; diagonal += (c - offset_i)
    dv_dr = np.outer(r_vec, dc_dr)
    dv_dr[0, 0] += c - 1.0
    dv_dr[1, 1] += c - 1.0
    dv_dr[2, 2] += c - 3.0

    G_J2 = (K / r5) * dv_dr - (5.0 * K / r7) * np.outer(v, r_vec)

    return a, G_J2


def earth_ekf_dynamics(t, x, et0, with_jacobian=True):
    """
    Earth-centred ECI dynamics for the pseudorange EKF.

    Forces: Earth gravity + J2 oblateness + Moon/Sun third-body (tidal) + SRP.
    State: [r (3 km) | v (3 km/s) | clkb (km) | clkdr (km/s) | gamma_srp (m^2/kg)]
    """
    r, v   = x[0:3], x[3:6]
    clkdr  = x[7]
    gamma  = x[8]

    et = et0 + t
    r_moon = ephemeris.get_lunar_pos(et)   # Moon ECI position  [km]
    r_sun  = ephemeris.get_sun_pos(et)     # Sun ECI position   [km]

    rho_moon = r - r_moon       # spacecraft - Moon
    rho_sun  = r - r_sun        # spacecraft - Sun

    r_norm        = np.linalg.norm(r)
    rho_moon_norm = np.linalg.norm(rho_moon)
    rho_sun_norm  = np.linalg.norm(rho_sun)
    r_moon_norm   = np.linalg.norm(r_moon)
    r_sun_norm    = np.linalg.norm(r_sun)

    # Earth central gravity
    a_earth = -MU_EARTH * r / r_norm ** 3

    # J2 oblateness (matches sim_infra/dynamics.py _get_j2_perturbation)
    a_j2, G_J2 = j2_accel_and_jacobian(r)

    # Moon third-body (tidal form removes Earth's acceleration toward Moon)
    a_moon  = MU_MOON * (-rho_moon / rho_moon_norm ** 3
                         - r_moon  / r_moon_norm  ** 3)

    # Sun third-body (same tidal form)
    a_sun   = MU_SUN  * (-rho_sun  / rho_sun_norm  ** 3
                         - r_sun   / r_sun_norm   ** 3)

    # SRP: pushes spacecraft away from the Sun
    a_srp   = SRP_COEFF * gamma * rho_sun / rho_sun_norm ** 3

    a_total = a_earth + a_j2 + a_moon + a_sun + a_srp

    dx_dt      = np.zeros(9)
    dx_dt[0:3] = v
    dx_dt[3:6] = a_total
    dx_dt[6]   = clkdr   # clkb_dot = clkdr
    dx_dt[7]   = 0.0     # clkdr treated as constant
    dx_dt[8]   = 0.0     # gamma treated as constant

    if not with_jacobian:
        return dx_dt, None

    I3 = np.eye(3)

    def _grav_tensor(u):
        """d(u/|u|^3)/du  -- gravity gradient matrix."""
        un = np.linalg.norm(u)
        return I3 / un ** 3 - 3.0 * np.outer(u, u) / un ** 5

    G_earth = _grav_tensor(r)
    G_moon  = _grav_tensor(rho_moon)
    G_sun   = _grav_tensor(rho_sun)

    J = np.zeros((9, 9))
    J[0:3, 3:6] = I3
    J[3:6, 0:3] = (
        -MU_EARTH * G_earth
        + G_J2
        - MU_MOON * G_moon
        - MU_SUN  * G_sun
        + SRP_COEFF * gamma * G_sun    # d(a_srp)/dr
    )
    J[3:6, 8] = SRP_COEFF * rho_sun / rho_sun_norm ** 3   # d(a_srp)/d(gamma)
    J[6, 7]   = 1.0                                         # d(clkb)/d(clkdr)

    return dx_dt, J


def make_thrust_dynamics(thrust_fn):
    """Wrap earth_ekf_dynamics with a continuous thrust acceleration function."""
    def _dynamics_with_thrust(t, x, et0, with_jacobian=True):
        dx_dt, J = earth_ekf_dynamics(t, x, et0, with_jacobian)
        dx_dt[3:6] = dx_dt[3:6] + thrust_fn(t, x)
        return dx_dt, J

    return _dynamics_with_thrust


def ekf_process_noise(sigma_acc, clk_q1, clk_q2, dt=1.0):
    """
    Discrete-time process noise Q for the 9-state EKF.
    Returns the full (9, 9) matrix; the UDU filter uses its diagonal.
    """
    Q   = np.zeros((9, 9))
    s2  = sigma_acc ** 2
    q   = np.array([[dt ** 3 / 3, dt ** 2 / 2],
                    [dt ** 2 / 2, dt          ]]) * s2
    for i in range(3):
        Q[i,     i    ] = q[0, 0] * 100
        Q[i,     i + 3] = q[0, 1] * 50
        Q[i + 3, i    ] = q[1, 0] * 50
        Q[i + 3, i + 3] = q[1, 1] * 10

    Q[6, 6] = clk_q1 * dt + clk_q2 * dt ** 3 / 3
    Q[6, 7] = clk_q2 * dt ** 2 / 2
    Q[7, 6] = clk_q2 * dt ** 2 / 2
    Q[7, 7] = clk_q2 * dt
    Q[8, 8] = 1e-16   # gamma: nearly constant, tiny diffusion

    return Q


def earth_ekf_dynamics_simple(x, r_sun, with_jacobian=True):
    """Simplified dynamics (Earth + J2 + SRP only). Caller provides pre-computed Sun ECI position [km]."""
    r, v  = x[0:3], x[3:6]
    clkdr = x[7]
    gamma = x[8]

    rho_sun      = r - r_sun
    rho_sun_norm = np.linalg.norm(rho_sun)
    r_norm       = np.linalg.norm(r)

    a_earth     = -MU_EARTH * r / r_norm ** 3
    a_j2, G_J2  = j2_accel_and_jacobian(r)
    a_srp       = SRP_COEFF * gamma * rho_sun / rho_sun_norm ** 3

    a_total = a_earth + a_j2 + a_srp

    dx_dt      = np.zeros(9)
    dx_dt[0:3] = v
    dx_dt[3:6] = a_total
    dx_dt[6]   = clkdr
    dx_dt[7]   = 0.0
    dx_dt[8]   = 0.0

    if not with_jacobian:
        return dx_dt, None

    I3 = np.eye(3)

    def _grav_tensor(u):
        un = np.linalg.norm(u)
        return I3 / un ** 3 - 3.0 * np.outer(u, u) / un ** 5

    G_earth = _grav_tensor(r)
    G_sun   = _grav_tensor(rho_sun)

    J = np.zeros((9, 9))
    J[0:3, 3:6] = I3
    J[3:6, 0:3] = (
        -MU_EARTH * G_earth
        + G_J2
        + SRP_COEFF * gamma * G_sun
    )
    J[3:6, 8] = SRP_COEFF * rho_sun / rho_sun_norm ** 3
    J[6, 7]   = 1.0

    return dx_dt, J


def make_cheby_dynamics(ephem, thrust_fn=None):
    """Full EKF dynamics using a pre-fitted ChebyEphemeris instead of per-step SPICE calls."""
    def _dyn(t, x, et0, with_jacobian=True):
        r, v   = x[0:3], x[3:6]
        clkdr  = x[7]
        gamma  = x[8]

        et     = et0 + t
        r_moon = ephem.get_moon_pos(et)
        r_sun  = ephem.get_sun_pos(et)

        rho_moon = r - r_moon
        rho_sun  = r - r_sun

        r_norm        = np.linalg.norm(r)
        rho_moon_norm = np.linalg.norm(rho_moon)
        rho_sun_norm  = np.linalg.norm(rho_sun)
        r_moon_norm   = np.linalg.norm(r_moon)
        r_sun_norm    = np.linalg.norm(r_sun)

        a_earth = -MU_EARTH * r / r_norm ** 3
        a_j2, G_J2 = j2_accel_and_jacobian(r)
        a_moon  = MU_MOON * (-rho_moon / rho_moon_norm ** 3
                             - r_moon  / r_moon_norm  ** 3)
        a_sun   = MU_SUN  * (-rho_sun  / rho_sun_norm  ** 3
                             - r_sun   / r_sun_norm   ** 3)
        a_srp   = SRP_COEFF * gamma * rho_sun / rho_sun_norm ** 3

        a_total = a_earth + a_j2 + a_moon + a_sun + a_srp

        dx_dt      = np.zeros(9)
        dx_dt[0:3] = v
        dx_dt[3:6] = a_total
        dx_dt[6]   = clkdr
        dx_dt[7]   = 0.0
        dx_dt[8]   = 0.0

        if thrust_fn is not None:
            dx_dt[3:6] += thrust_fn(t, x)

        if not with_jacobian:
            return dx_dt, None

        I3 = np.eye(3)

        def _grav_tensor(u):
            un = np.linalg.norm(u)
            return I3 / un ** 3 - 3.0 * np.outer(u, u) / un ** 5

        G_earth = _grav_tensor(r)
        G_moon  = _grav_tensor(rho_moon)
        G_sun   = _grav_tensor(rho_sun)

        J = np.zeros((9, 9))
        J[0:3, 3:6] = I3
        J[3:6, 0:3] = (
            -MU_EARTH * G_earth
            + G_J2
            - MU_MOON * G_moon
            - MU_SUN  * G_sun
            + SRP_COEFF * gamma * G_sun
        )
        J[3:6, 8] = SRP_COEFF * rho_sun / rho_sun_norm ** 3
        J[6, 7]   = 1.0

        return dx_dt, J

    return _dyn
