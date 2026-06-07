import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from scipy.interpolate import interp1d

import AA278.project.misc.utils as utils
from AA278.project.misc.utils import COE
from AA278.project.misc.constants import DAY_TO_SEC, R_EARTH, MU_EARTH, MIN_TO_SEC
import AA278.project.orbital.ephemeris as ephemeris
import AA278.project.sim_infra.satellite as satellite

from AA278.project.orbital.trajectory_planner import ContinuousBurnTrajectoryPlan, ProgradeLaw, ProgradeNormalLaw
from AA278.project.orbital.mission_planner import MissionPlanner
from AA278.project.sensors.gnss_measurements import run_gnss_time_history
from AA278.project.nav.cheby_ephemeris import ChebyEphemeris
from AA278.project.nav.ekf_dynamics import (
    make_cheby_dynamics, ekf_process_noise, CLK_Q1, CLK_Q2,
)
from AA278.project.nav.pseudorange_filter import run_pseudorange_filter
from AA278.project.nav.udu_filter import UDUFilter

DATA_DIR = "./AA278/project/spice_kernels"

_G0_KM_S2 = 9.80665e-3   # standard gravity [km/s^2]


def _make_spiral_thrust_fn(thrust_N, isp_s, m0_kg, burn_start_s, burn_duration, direction_law):
    """Build thrust acceleration function for the spiral EKF dynamics."""
    # kg/s; g0 in km/s^2 -> use SI: mdot = F[N] / (g0[m/s^2] * Isp[s])
    mdot = thrust_N / (_G0_KM_S2 * 1e3 * isp_s)   # [kg/s]

    def _thrust_fn(t_rel, x9):
        burn_end = burn_start_s + burn_duration
        if t_rel < burn_start_s or t_rel > burn_end:
            return np.zeros(3)
        m = max(m0_kg - mdot * (t_rel - burn_start_s), 1.0)
        a_mag = thrust_N / m * 1e-3              # N/kg -> m/s^2 -> km/s^2
        return a_mag * direction_law.direction(t_rel, x9)

    return _thrust_fn


# ---------------------------------------------------------------------------
# Performance plot
# ---------------------------------------------------------------------------

def _plot_nav_performance(times, states, filter_times, x_hist, P_hist,
                           t_burn=None, title="Pseudorange UDU Filter"):
    # Interpolate true trajectory to filter epochs
    true_r = interp1d(times, states[:3],   axis=1, assume_sorted=True)(filter_times).T
    true_v = interp1d(times, states[3:6],  axis=1, assume_sorted=True)(filter_times).T

    pos_err = (x_hist[:, :3]  - true_r) * 1e3    # km -> m
    vel_err = (x_hist[:, 3:6] - true_v) * 1e3    # km/s -> m/s

    pos_rmse = np.sqrt(np.mean(pos_err**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_err**2, axis=0))
    print(f"\n--- Navigation Filter RMSE ({title}) ---")
    print(f"  Position:  x={pos_rmse[0]:.2f} m,  y={pos_rmse[1]:.2f} m,  z={pos_rmse[2]:.2f} m")
    print(f"  Velocity: vx={vel_rmse[0]:.4f} m/s, vy={vel_rmse[1]:.4f} m/s, vz={vel_rmse[2]:.4f} m/s")
    print(f"  Total position RMSE : {np.sqrt(np.sum(pos_rmse**2)):.2f} m")
    print(f"  Total velocity RMSE : {np.sqrt(np.sum(vel_rmse**2)):.4f} m/s")

    # 1-sigma from the diagonal of the filter covariance [m]
    pos_sig = np.sqrt(np.maximum(
        np.diagonal(P_hist[:, :3, :3],   axis1=1, axis2=2), 0)) * 1e3
    vel_sig = np.sqrt(np.maximum(
        np.diagonal(P_hist[:, 3:6, 3:6], axis1=1, axis2=2), 0)) * 1e3

    t_days  = filter_times / DAY_TO_SEC
    labels  = [('x', 'vx'), ('y', 'vy'), ('z', 'vz')]
    colors  = ['steelblue', 'seagreen', 'darkorange']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle(title, fontsize=13)

    for i, (lp, lv) in enumerate(labels):
        # Position row
        ax = axes[0, i]
        ax.plot(t_days, pos_err[:, i], color=colors[i], linewidth=0.7, label='error')
        ax.fill_between(t_days,
                        -3 * pos_sig[:, i],  3 * pos_sig[:, i],
                        alpha=0.25, color=colors[i], label='3sigma')
        ax.axhline(0, color='k', linewidth=0.4)
        if t_burn is not None:
            t_burn_day = t_burn / DAY_TO_SEC
            ax.axvline(t_burn_day, color='red', linewidth=0.8, linestyle='--',
                       label='burn' if i == 0 else None)
        ax.set_ylabel(f"{lp} error  [m]")
        ax.set_ylim(-400, 400)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linewidth=0.4, alpha=0.5)

        # Velocity row
        ax = axes[1, i]
        ax.plot(t_days, vel_err[:, i], color=colors[i], linewidth=0.7, label='error')
        ax.fill_between(t_days,
                        -3 * vel_sig[:, i],  3 * vel_sig[:, i],
                        alpha=0.25, color=colors[i], label='3sigma')
        ax.axhline(0, color='k', linewidth=0.4)
        if t_burn is not None:
            t_burn_day = t_burn / DAY_TO_SEC
            ax.axvline(t_burn_day, color='red', linewidth=0.8, linestyle='--')
        ax.set_ylabel(f"{lv} error  [m/s]")
        ax.set_xlabel("Mission time  [days]")
        ax.set_ylim(-1, 1)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.show()
    return fig, axes


def main(plot_sun=False, plot_moon=False, animate=False):
    # Sim setup
    t0 = 0
    num_days = 39.0
    # num_days = 0.5
    duration = num_days * DAY_TO_SEC
    print("Running simulation for " + str(num_days) + " days.")

    # Load SPICE Kernels
    ephemeris.load_kernels(DATA_DIR)
    et0 = spice.str2et("2026-04-15T08:00:00")
    parking_orbit_alt_km = 500 #km

    # Set initial orbit
    init_coe = COE(
        sma= R_EARTH + parking_orbit_alt_km,
        ecc=0.0,
        inc=np.deg2rad(0),
        raan=0.,
        arg_peri=0.,
        mean_anom=0.,
    )

    sat = satellite.Satellite(mass_kg=50.0)
    sat.initialize_from_coe(init_coe, MU_EARTH)
    initial_state = sat.get_state()

    # pure prograde law
    direction_law = ProgradeLaw()

    direction_law = ProgradeNormalLaw(alpha_deg=15.0)

    trajectory_plan = ContinuousBurnTrajectoryPlan(
        burn_start_time=15.5 * 60,
        burn_duration=35. * DAY_TO_SEC,
        thrust_N=0.1,
        isp_s=4000,
        direction_law=direction_law,
    )
    phases = trajectory_plan.make_plan(x0=initial_state, t0=t0, tf=t0+duration)

    mission_sim = MissionPlanner(sat=sat, et0=et0)
    results = mission_sim.run_trajectory(
        t0=t0,
        x0=initial_state,
        traj_plan=phases,
    )

    times = results.t
    et_time = et0 + times
    moon_pos_time_hist = ephemeris.get_lunar_pos(et_time)
    sun_pos_time_hist = ephemeris.get_sun_pos(et_time)
    states = results.x

    # if animate:
        # utils.animate_trajectory(
        #     states=states,
        #     time=times,
        #     moon_pos_time_hist=moon_pos_time_hist,
        #     R_EARTH=R_EARTH,
        #     interval=50,
        #     skip=5000,
        #     save_path="spiral_animation.gif",
        # )
        # plt.plot(times, states[6])
        # plt.title("Spacecraft Mass")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Spacecraft mass (kg)")
        # plt.grid()
        # plt.show()
    # else:
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")

        # ax.plot(states[0], states[1], states[2], label="Orbit")
        # ax.plot(
        #     moon_pos_time_hist[:, 0],
        #     moon_pos_time_hist[:, 1],
        #     moon_pos_time_hist[:, 2],
        #     label="Moon",
        # )

        # utils.plot_earth(ax, R_EARTH=R_EARTH)

        # ax.set_xlabel("x [km]")
        # ax.set_ylabel("y [km]")
        # ax.set_zlabel("z [km]")
        # ax.legend()

        # utils.set_axes_equal(ax)
        # plt.show()
    print("sim finished!")
    # GNSS post-processing at 1 Hz (dual-frequency)
    gnss = run_gnss_time_history(times, states, et0, sample_rate_hz=1/10)
    print("Gnss measurements made")
    # gnss.plot_overview()

    # Pseudorange navigation filter
    gamma_true = sat.cr * sat.surf_area_m2 / sat.mass_kg
    x0_true = np.array([*states[:6, 0], 0.0, 0.0, gamma_true])
    sigma_init = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 0.1, 1e-6, gamma_true * 0.5])
    rng = np.random.default_rng(42)
    x0_filter = x0_true + rng.standard_normal(9) * sigma_init
    P0 = np.diag(sigma_init**2)

    # Fit Chebyshev ephemeris for Moon and Sun over the full mission window
    ephem = ChebyEphemeris.build(et0, et0 + duration)

    # Build thrust-aware EKF dynamics using the exact scenario parameters
    thrust_fn = _make_spiral_thrust_fn(
        thrust_N      = trajectory_plan.thrust_N,
        isp_s         = trajectory_plan.isp_s,
        m0_kg         = sat.mass_kg,
        burn_start_s  = trajectory_plan.burn_start_time,
        burn_duration = trajectory_plan.burn_duration,
        direction_law = direction_law,
    )
    thrust_dynamics = make_cheby_dynamics(ephem, thrust_fn)

    print("  Running pseudorange filter (full length)...")
    filter_times, x_hist, P_hist = run_pseudorange_filter(
        gnss=gnss, et0=et0, x0=x0_filter, P0=P0,
        dynamics_fn=thrust_dynamics,
        udu_filter_cls=UDUFilter,
        process_noise_fn=ekf_process_noise,
        sigma_acc=1e-6,
        clk_q1=CLK_Q1, clk_q2=CLK_Q2,
        # predict_dt_s=1.0,
    )
    print(f"  Filter complete: {len(filter_times):,} epochs.")
    
    _plot_nav_performance(times, states, filter_times, x_hist, P_hist,
                           title="Position/Velocity Estimate Spiral Transfer")

    gnss.plot_nav_analysis()

    moon_pos_filter = ephemeris.get_lunar_pos(et0 + filter_times)
    utils.plot_synodic_covariance(filter_times, P_hist, moon_pos_filter,
                                  title="Position Covariance - Synodic Frame (Spiral Transfer)")


if __name__ == "__main__":
    # main(animate=True)
    main(animate=False)
