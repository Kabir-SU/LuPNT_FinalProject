"""
Pseudorange UDU navigation filter.
Measurement model: z = |r_rx - r_sv| + clkb + noise
No carrier phase, no cross-epoch caching -- simpler and more robust than TDCP.
"""
import numpy as np
import tqdm
from scipy.integrate import solve_ivp
from AA278.project.sensors.gnss_sensor import SIGNAL_BANDS


class PseudorangeEKFProblem:
    """UDUFilter adapter for pseudorange-based EKF.

    State: x = [r (3 km) | v (3 km/s) | clkb (km) | clkdr (km/s) | gamma_srp]
    """
    n_state = 9

    def __init__(self, et0: float, dynamics_fn, constellation=None):
        self.et0 = et0
        self.dynamics_fn = dynamics_fn
        self.constellation = constellation
        self.r_tx_now   = np.zeros((0, 3))
        self.sigmas_now = np.array([], dtype=float)
        self.n_meas     = 0

    def set_measurements(self, r_tx, pseudorange_sigmas):
        """Load per-epoch data before UDUFilter.update().
        r_tx: (N, 3) SV ECI positions [km]
        pseudorange_sigmas: (N,) 1-sigma noise [km]
        """
        self.r_tx_now   = np.asarray(r_tx, dtype=float)
        self.sigmas_now = np.asarray(pseudorange_sigmas, dtype=float)
        self.n_meas     = len(pseudorange_sigmas)

    def _dynamics_ode_with_stm(self, t, y):
        n        = self.n_state
        x_state  = y[:n]
        stm_flat = y[n:]
        dydt     = np.zeros_like(y)
        Phi      = stm_flat.reshape((n, n))
        dxdt, J  = self.dynamics_fn(t, x_state, self.et0, with_jacobian=True)
        dydt[:n] = dxdt
        dydt[n:] = (J @ Phi).reshape(n * n)
        return dydt

    def dynamics(self, t, x, dt):
        n  = self.n_state
        y0 = np.hstack((x, np.eye(n).reshape(n * n)))
        sol = solve_ivp(
            self._dynamics_ode_with_stm,
            t_span=(t, t + dt),
            y0=y0,
            method='RK45',
            rtol=1e-7,
            atol=1e-10,
        )
        yf     = sol.y[:, -1]
        x_next = yf[:n]
        F      = yf[n:].reshape((n, n))
        return x_next, F

    def predicted_measurement(self, t, x, k):
        r_sv  = self.r_tx_now[k]
        r_rx  = x[:3]
        diff  = r_rx - r_sv
        rho   = np.linalg.norm(diff)
        z_pred = rho + x[6]           # range + clock bias
        H      = np.zeros(self.n_state)
        H[:3]  = diff / rho            # d(rho)/d(r)
        H[6]   = 1.0                   # d(z)/d(clkb)
        return z_pred, H

    def measurement_noise(self, k):
        return float(self.sigmas_now[k])**2


def _inflate_velocity_covariance(udu, sigma_dv_sq: float) -> None:
    """Add sigma_dv_sq to each velocity diagonal of P via a Thornton MWGS update."""
    n = udu.problem.n_state
    W = np.hstack((udu.U.copy(), np.eye(n)))
    q_burn = np.zeros(n)
    q_burn[3:6] = sigma_dv_sq
    d_tilde = np.hstack((udu.d.copy(), q_burn))

    U_new = np.eye(n)
    d_new = np.zeros(n)
    for j in range(n - 1, -1, -1):
        d_new[j] = float(d_tilde @ (W[j] ** 2))
        for i in range(j):
            U_new[i, j] = float((d_tilde * W[i]) @ W[j]) / d_new[j]
            W[i] -= U_new[i, j] * W[j]
    udu.U = U_new
    udu.d = d_new


def run_pseudorange_filter(
    gnss,
    et0: float,
    x0: np.ndarray,
    P0: np.ndarray,
    dynamics_fn,
    udu_filter_cls,
    process_noise_fn,
    *,
    min_sats: int          = 4,
    sigma_acc: float       = 1e-7,
    clk_q1: float,
    clk_q2: float,
    burn_time=None,
    burn_dv=None,
    burn_sigma_dv: float = 0.01,
    predict_dt_s: float | None = None,
    max_epochs=None,
):
    """Run pseudorange UDU EKF over a GNSSTimeHistory. Returns (times, x_hist, P_hist)."""
    bp  = SIGNAL_BANDS['L1']
    psr = gnss.pseudorange_L1
    cnr = gnss.cnr_L1.astype(np.float64)
    vis = gnss.tracked_L1()

    N_epochs = len(gnss.times_s)
    if max_epochs is not None:
        N_epochs = min(N_epochs, max_epochs + 1)

    # Burn epoch: predict TO t=burn_time, apply DeltaV, then propagate past
    burn_epoch = None
    if burn_time is not None and burn_dv is not None:
        burn_epoch = int(np.searchsorted(gnss.times_s, burn_time, side='left'))
        print(f"  Burn at t={burn_time:.1f} s -> epoch {burn_epoch} "
              f"(t={gnss.times_s[burn_epoch]:.1f} s)")

    problem = PseudorangeEKFProblem(et0=et0, dynamics_fn=dynamics_fn,
                                     constellation=gnss.constellation)
    udu     = udu_filter_cls(problem, x0, P0, np.zeros(9))

    times   = []
    x_hist  = []
    P_hist  = []

    for k in tqdm.tqdm(range(1, N_epochs)):
        t_rel = float(gnss.times_s[k])
        dt    = float(gnss.times_s[k] - gnss.times_s[k - 1])

        # Time update - optionally sub-stepped for better STM linearisation
        if predict_dt_s is not None and predict_dt_s < dt - 1e-9:
            # Compose multiple short predict steps between measurement epochs.
            # This is more accurate on eccentric orbits where the 1-step STM
            # over a long interval is poorly linearised.
            n_sub   = max(1, int(round(dt / predict_dt_s)))
            dt_sub  = dt / n_sub
            t_start = t_rel - dt
            for s in range(n_sub):
                udu.q = np.diag(process_noise_fn(sigma_acc, clk_q1, clk_q2, dt=dt_sub))
                udu.predict(t_start + s * dt_sub, dt_sub)
        else:
            udu.q = np.diag(process_noise_fn(sigma_acc, clk_q1, clk_q2, dt=dt))
            udu.predict(t_rel - dt, dt)

        # Impulsive burn: apply nominal DeltaV to state mean, then inflate the
        # velocity covariance by the DeltaV execution uncertainty so post-burn
        # pseudoranges can quickly correct any pre-existing velocity error.
        if burn_epoch is not None and k == burn_epoch:
            udu.x[3:6] = udu.x[3:6] + np.asarray(burn_dv)
            if burn_sigma_dv > 0.0:
                _inflate_velocity_covariance(udu, burn_sigma_dv ** 2)

        # Select visible satellites at this epoch
        vis_k   = vis[k]
        sat_ids = np.where(vis_k)[0]

        if len(sat_ids) >= min_sats:
            # Per-satellite pseudorange sigma from C/N0
            cnr_k     = cnr[k, sat_ids]
            sig_k     = bp['sigma_ref_km'] * 10.0**((bp['cnr_ref_db_hz'] - cnr_k) / 20.0)
            r_tx_k    = gnss.sv_pos[k, sat_ids]   # (N, 3)
            z_meas    = psr[k, sat_ids]            # (N,)

            problem.set_measurements(r_tx=r_tx_k, pseudorange_sigmas=sig_k)
            udu.update(t_rel, z_meas)

        times.append(t_rel)
        x_hist.append(udu.x.copy())
        P_hist.append(udu.P())

    return np.array(times), np.array(x_hist), np.array(P_hist)
