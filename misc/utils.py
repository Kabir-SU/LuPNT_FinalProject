import numpy as np
from numpy.typing import NDArray
import AA278.project.misc.constants as constants
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wrap_to_2pi(angle):
    """
    Wraps an angle in radians to the range [0, 2*pi)
    """
    return angle % (2 * np.pi)

def wrap_to_pi(angle):
    """
    Wraps an angle in radians to the range [-pi, pi)
    """
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped_angle

def get_E(M, e, eps=1e-8, max_iter=100):
    """
    Implements Newton-Raphson method to obtain E
    M in rad, e is unitless
    """
    M = wrap_to_2pi(M)
    E_cur = np.pi
    err_store = np.zeros(max_iter+1)
    err = np.inf
    iter_num = 1

    err_store[0] = err

    for i in range(max_iter): 
        E_cur = E_cur - (E_cur - e * np.sin(E_cur) - M) / (1 - e * np.cos(E_cur))
        err = E_cur - e * np.sin(E_cur) - M
        err_store[i+1] = err

        if abs(err) < eps:
            iter_num = i+1
            break

    E_cur = wrap_to_2pi(E_cur)

    return E_cur

def get_rv_PQW(a, e, E, mu):
    """
    Get r and v in PQW frame given a, e, E
    a in km, e is unitless, E in rad
    """
    r_PQW = np.zeros(3)  # dummy
    v_PQW = np.zeros(3)  # dummy

    # add your code here
    n = np.sqrt(mu / a**3)
    r_PQW = np.array([a * (np.cos(E) - e), a * np.sqrt(1 - e**2) * np.sin(E), 0])
    v_PQW = a * n / (1 - e * np.cos(E)) * np.array([
        -np.sin(E),
        np.sqrt(1 - e**2) * np.cos(E),
        0.0
    ])

    return r_PQW, v_PQW

def get_rot_PQW_to_IJK(inc, raan, omega):
    """
    Get rotation matrix from PQW to IJK frame given inc, raan, omega
    inc, raan, omega in rad
    """
    rot_PQW_to_IJK = np.zeros((3, 3)) # dummy

    c_omega = np.cos(omega)
    s_omega = np.sin(omega)
    c_raan = np.cos(raan)
    s_raan = np.sin(raan)
    c_inc = np.cos(inc)
    s_inc = np.sin(inc)

    rot_PQW_to_IJK = np.array([
        [c_raan * c_omega - s_raan * s_omega * c_inc, -c_raan * s_omega - s_raan * c_omega * c_inc, s_raan * s_inc],
        [s_raan * c_omega + c_raan * s_omega * c_inc, -s_raan * s_omega + c_raan * c_omega * c_inc, -c_raan * s_inc],
        [s_omega * s_inc, c_omega * s_inc, c_inc]
    ])

    return rot_PQW_to_IJK

def coe_to_cart(coe: COE, mu):
    """
    Get cartesian state in inertial frame given coe
    coe = (a, e, inc, raan, omega, M)
    a in km, e is unitless, inc, raan, omega, M in rad
    """
    a = coe.sma
    e = coe.ecc
    inc = coe.inc
    raan = coe.raan
    omega = coe.arg_peri
    M = coe.mean_anom

    # a, e, inc, raan, omega, M = coe
    rv_cart = np.zeros(6)  # dummy

    E = get_E(M=M, e=e)
    r_PQW, v_PQW = get_rv_PQW(a=a, e=e, E=E, mu=mu)
    rot_PQW_to_IJK = get_rot_PQW_to_IJK(inc=inc, raan=raan, omega=omega)
    r_inertial = rot_PQW_to_IJK @ r_PQW
    v_inertial = rot_PQW_to_IJK @ v_PQW
    rv_cart[:3] = r_inertial
    rv_cart[3:6] = v_inertial
    return rv_cart


def cart_to_coe(cartposvel: CartPosVel, mu):
    """
    Get coe given cartesian state in inertial frame
    cart_x = (r_vec, v_vec)
    r_vec in km, v_vec in km/s
    """
    a = 0  # dummy
    e = 0  # dummy
    inc = 0  # dummy
    raan = 0  # dummy
    omega = 0  # dummy
    M = 0  # dummy

    r_vec = cartposvel.pos
    v_vec = cartposvel.pos
    r_norm = np.linalg.norm(r_vec)
    v_norm = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    k_vec = np.array([0, 0, 1])
    n_vec = np.cross(k_vec, h_vec)
    e_vec = np.cross(v_vec, h_vec) / mu - r_vec / r_norm
    e = np.linalg.norm(e_vec)

    xi = v_norm**2 / 2 - mu / r_norm
    a = -mu / (2 * xi)
    p = a * (1 - e**2)

    cos_i = h_vec[2] / h_norm
    cos_Omega = n_vec[0] / np.linalg.norm(n_vec)
    cos_w = (n_vec @ e_vec) / (np.linalg.norm(n_vec) * e)
    cos_nu = (e_vec @ r_vec) / (r_norm * e)

    inc = np.arccos(cos_i)
    raan = np.arccos(cos_Omega)
    omega = np.arccos(cos_w)
    nu = np.arccos(np.clip(cos_nu, -1, 1))

    if n_vec[1] < 0:
        raan = 2 * np.pi - raan
    if e_vec[2] < 0:
        omega = 2 * np.pi - omega
    if (r_vec @ v_vec) < 0:
        nu = 2 * np.pi - nu

    den = 1 + e * np.cos(nu)
    sinE = np.sqrt(1 - e**2) * np.sin(nu) / den
    cosE = (e + np.cos(nu)) / den
    E = np.arctan2(sinE, cosE)
    E = wrap_to_2pi(E)
    M = E - e * np.sin(E)

    # wrap angles to the ranges
    inc = wrap_to_pi(inc)
    raan = wrap_to_2pi(raan)
    omega = wrap_to_2pi(omega)
    M = wrap_to_2pi(M)

    coe = COE(sma=a, ecc=e, inc=inc, raan=raan, arg_peri=omega, mean_anom=M)

    return coe

@dataclass
class COE:
    """Dataclass storing classical orbital elements"""
    sma: float
    ecc: float
    inc: float
    raan: float
    arg_peri: float
    mean_anom: float

@dataclass
class CartPosVel:
    """Dataclass storing cartesian positions and velocities"""
    pos: NDArray[np.float]
    vel: NDArray[np.float]


def plot_earth(ax, R_EARTH=constants.R_EARTH, n=60):
    """Add Earth as a sphere centered at the origin."""

    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)

    x = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, alpha=0.35, linewidth=0)


def set_axes_equal(ax):
    """Make 3D axes have equal scale so Earth/orbits are not visually distorted."""

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    max_range = 0.5 * max(
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0],
    )

    ax.set_xlim3d(x_mid - max_range, x_mid + max_range)
    ax.set_ylim3d(y_mid - max_range, y_mid + max_range)
    ax.set_zlim3d(z_mid - max_range, z_mid + max_range)

def post_process_integration(solution, event_names):
    if solution.success:
        time_hist = solution.t
        state_time_hist = solution.y

        for name, t_event, y_event in zip(event_names, solution.t_events, solution.y_events):
            if len(t_event) > 0:
                t_hit = t_event[0]
                print(f"{name} detected at t = {t_hit / 86400.0:.3f} days")

        return time_hist, state_time_hist
    else:
        return ValueError("Orbit Propagation Failed!")
    

def animate_trajectory(
    states,
    time=None,
    moon_pos_time_hist=None,
    R_EARTH=6378.1363,
    interval=50,
    skip=100,
    save_path=None,
):
    r_sc = states[0:3, ::skip]

    if time is not None:
        time_anim = time[::skip]
    else:
        time_anim = None

    if moon_pos_time_hist is not None:
        r_moon = moon_pos_time_hist[::skip, :]
    else:
        r_moon = None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(r_sc[0], r_sc[1], r_sc[2], alpha=0.25, label="Spacecraft path")

    if r_moon is not None:
        ax.plot(r_moon[:, 0], r_moon[:, 1], r_moon[:, 2], alpha=0.25, label="Moon path")

    plot_earth(ax, R_EARTH=R_EARTH)

    sc_point, = ax.plot([], [], [], marker="o", linestyle="", label="Spacecraft")
    sc_trace, = ax.plot([], [], [], linewidth=1.5)

    if r_moon is not None:
        moon_point, = ax.plot([], [], [], marker="o", linestyle="", label="Moon")
    else:
        moon_point = None

    time_text = ax.text2D(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
    )

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend()

    # ... keep your axis limit code unchanged ...

    def update(frame):
        sc_point.set_data([r_sc[0, frame]], [r_sc[1, frame]])
        sc_point.set_3d_properties([r_sc[2, frame]])

        sc_trace.set_data(r_sc[0, :frame + 1], r_sc[1, :frame + 1])
        sc_trace.set_3d_properties(r_sc[2, :frame + 1])

        if time_anim is not None:
            elapsed_days = time_anim[frame] / 86400.0
            time_text.set_text(f"Elapsed Time: {elapsed_days:.2f} days")
        else:
            time_text.set_text(f"Frame: {frame}")

        artists = [sc_point, sc_trace, time_text]

        if moon_point is not None:
            moon_point.set_data([r_moon[frame, 0]], [r_moon[frame, 1]])
            moon_point.set_3d_properties([r_moon[frame, 2]])
            artists.append(moon_point)

        return artists

    anim = FuncAnimation(
        fig,
        update,
        frames=r_sc.shape[1],
        interval=interval,
        blit=False,
    )

    if save_path is not None:
        anim.save(save_path)

    plt.show()

    return anim


def plot_synodic_covariance(
    filter_times: np.ndarray,
    P_hist: np.ndarray,
    moon_pos: np.ndarray,
    day_axis: bool = True,
    title: str = "Position Covariance - Synodic (Rotating) Frame",
    t_burn: float | None = None,
):
    """Plot 1-sigma position uncertainty in the Earth-Moon synodic rotating frame (radial/tangential/normal)."""
    N = len(filter_times)
    t = filter_times / 86400.0 if day_axis else filter_times
    xlabel = "Mission time  [days]" if day_axis else "Mission time  [s]"

    # Moon velocity via central finite differences
    v_moon = np.gradient(moon_pos, filter_times, axis=0)   # (N, 3) km/s

    sigma_rtn = np.full((N, 3), np.nan)

    for k in range(N):
        rm = moon_pos[k]
        rm_norm = np.linalg.norm(rm)
        if rm_norm == 0.0:
            continue

        x_hat = rm / rm_norm

        h = np.cross(rm, v_moon[k])
        h_norm = np.linalg.norm(h)
        if h_norm == 0.0:
            continue
        z_hat = h / h_norm

        y_hat = np.cross(z_hat, x_hat)
        y_norm = np.linalg.norm(y_hat)
        if y_norm == 0.0:
            continue
        y_hat /= y_norm

        T = np.vstack([x_hat, y_hat, z_hat])          # (3, 3) ECI to synodic
        P_pos = P_hist[k, :3, :3]
        P_syn = T @ P_pos @ T.T
        diag_syn = np.diag(P_syn)
        if np.any(diag_syn < 0):
            continue
        sigma_rtn[k] = np.sqrt(diag_syn) * 1e3        # km to m

    labels = ['Radial (r)', 'Tangential (t)', 'Normal (n)']
    colors = ['steelblue', 'seagreen', 'darkorange']

    t_burn_day = t_burn / 86400.0 if (t_burn is not None and day_axis) else t_burn

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=13)

    for i, (label, color) in enumerate(zip(labels, colors)):
        ax = axes[i]
        ax.fill_between(t, 0, sigma_rtn[:, i], alpha=0.25, color=color)
        ax.plot(t, sigma_rtn[:, i], color=color, linewidth=0.8, label=f'1-sigma {label}')
        if t_burn is not None:
            ax.axvline(t_burn_day, color='red', linewidth=0.8, linestyle='--',
                       label='burn' if i == 0 else None)
        ax.set_ylabel(f"sigma_{label[0].lower()}  [m]")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()
    return fig, axes