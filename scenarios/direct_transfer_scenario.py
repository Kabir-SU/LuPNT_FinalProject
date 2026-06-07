import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
from scipy.interpolate import interp1d

import AA278.project.misc.utils as utils
from AA278.project.misc.utils import COE
from AA278.project.misc.constants import (
    DAY_TO_SEC, R_EARTH, MU_EARTH, MIN_TO_SEC,
)
import AA278.project.orbital.ephemeris as ephemeris
import AA278.project.sim_infra.satellite as satellite

from AA278.project.orbital.trajectory_planner import ImpulsiveTrajectoryPlan
from AA278.project.orbital.mission_planner import MissionPlanner
from AA278.project.sensors.gnss_measurements import run_gnss_time_history
from AA278.project.nav.udu_filter import UDUFilter
from AA278.project.nav.ekf_dynamics import (
    earth_ekf_dynamics, ekf_process_noise, CLK_Q1, CLK_Q2,
)
from AA278.project.nav.pseudorange_filter import run_pseudorange_filter

DATA_DIR = "./AA278/project/spice_kernels"

# Performance plot

def _plot_nav_performance(times, states, filter_times, x_hist, P_hist, t_burn,
                          title="Pseudorange UDU Filter"):
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
    t_burn_day = t_burn / DAY_TO_SEC
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
        ax.axvline(t_burn_day, color='red', linewidth=0.8, linestyle='--',
                   label='burn' if i == 0 else None)
        ax.set_ylabel(f"{lp} error  [m]")
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
        ax.axvline(t_burn_day, color='red', linewidth=0.8, linestyle='--')
        ax.set_ylabel(f"{lv} error  [m/s]")
        ax.set_xlabel("Mission time  [days]")
        ax.set_ylim(-0.1, 0.1)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.show()
    return fig, axes


# ---------------------------------------------------------------------------
# Main scenario
# ---------------------------------------------------------------------------

def main(animate=False):
    # Sim setup
    t0 = 0
    num_days = 5.2
    duration = num_days * DAY_TO_SEC
    print("Running simulation for " + str(num_days) + " days.")

    # Load SPICE Kernels
    ephemeris.load_kernels(DATA_DIR)
    et0 = spice.str2et("2026-04-25T08:00:00")
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

    orbit_period = 2 * np.pi * np.sqrt((R_EARTH + parking_orbit_alt_km)**3 / MU_EARTH)
    print("Initial Orbital Period in minutes: ", orbit_period / MIN_TO_SEC)

    sat = satellite.Satellite()
    sat.initialize_from_coe(init_coe, MU_EARTH)
    initial_state = sat.get_state()

    t_burn        = 15.5 * 60   # impulsive burn epoch [s]
    burn_duration = 5. * DAY_TO_SEC

    trajectory_plan = ImpulsiveTrajectoryPlan(
        t_burn=t_burn,
        burn_duration=burn_duration,
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
    states = results.x   # (7, N_t): rows = [x, y, z, vx, vy, vz, mass]

    # if animate:
    #     utils.animate_trajectory(
    #         states=states,
    #         time=times,
    #         moon_pos_time_hist=moon_pos_time_hist,
    #         R_EARTH=R_EARTH,
    #         interval=50,
    #         skip=1000,
    #         save_path="high_dv_animation.gif",
    #     )

    # else:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")

    #     ax.plot(states[0], states[1], states[2], label="Orbit")
    #     ax.plot(
    #         moon_pos_time_hist[:, 0],
    #         moon_pos_time_hist[:, 1],
    #         moon_pos_time_hist[:, 2],
    #         label="Moon",
    #     )

    #     utils.plot_earth(ax, R_EARTH=R_EARTH)

    #     ax.set_xlabel("x [km]")
    #     ax.set_ylabel("y [km]")
    #     ax.set_zlabel("z [km]")
    #     ax.legend()

    #     utils.set_axes_equal(ax)
    #     plt.show()

    # -----------------------------------------------------------------------
    # GNSS post-processing at 1 Hz (dual-frequency)
    # -----------------------------------------------------------------------
    gnss = run_gnss_time_history(times, states, et0, sample_rate_hz=1/10)
    # gnss.plot_overview()
    gnss.plot_nav_analysis()

    # -----------------------------------------------------------------------
    # Pseudorange navigation filter
    # -----------------------------------------------------------------------

    # Extract burn DeltaV from the velocity discontinuity in the state history.
    # The impulsive burn is not written into the time history, so the results
    # contain two states at the same t_burn: the last epoch of the pre-burn
    # coast and the first epoch of the post-burn coast.
    burn_candidates = np.where(np.abs(times - t_burn) < 0.5)[0]
    if len(burn_candidates) >= 2:
        burn_dv_eci = states[3:6, burn_candidates[1]] - states[3:6, burn_candidates[0]]
    else:
        # Fallback: no discontinuity found (shouldn't happen with a 1 s grid)
        burn_dv_eci = np.zeros(3)
        print("  Warning: could not locate velocity discontinuity at t_burn.")
    print(f"  Filter burn DeltaV: {np.linalg.norm(burn_dv_eci)*1e3:.1f} m/s")

    # Initial filter state: true state + representative initialisation errors
    gamma_true = sat.cr * sat.surf_area_m2 / sat.mass_kg   # m^2/kg
    x0_true = np.array([
        *states[:6, 0],   # r [km], v [km/s]
        0.0,               # clock bias [km]
        0.0,               # clock drift [km/s]
        gamma_true,        # SRP coefficient [m^2/kg]
    ])

    # Perturbation magnitudes
    sigma_init = np.array([
        1.0,  1.0,  1.0,          # position 1 km per axis
        1e-3, 1e-3, 1e-3,         # velocity 1 m/s per axis
        0.1,                       # clock bias 100 m
        1e-6,                      # clock drift 1 mm/s
        gamma_true * 0.5,          # gamma 50% relative error
    ])
    rng = np.random.default_rng(42)
    x0_filter = x0_true + rng.standard_normal(9) * sigma_init
    P0        = np.diag(sigma_init ** 2)

    print("  Running full pseudorange filter...")
    filter_times, x_hist, P_hist = run_pseudorange_filter(
        gnss=gnss, et0=et0, x0=x0_filter, P0=P0,
        dynamics_fn=earth_ekf_dynamics,
        udu_filter_cls=UDUFilter,
        process_noise_fn=ekf_process_noise,
        sigma_acc=1e-7, clk_q1=CLK_Q1, clk_q2=CLK_Q2,
        burn_time=t_burn, burn_dv=burn_dv_eci, burn_sigma_dv=0.01,
    )
    _plot_nav_performance(times, states, filter_times, x_hist, P_hist, t_burn,
                          title="Position/Velocity Estimation Direct Lunar Transfer")
    moon_pos_filter = ephemeris.get_lunar_pos(et0 + filter_times)
    utils.plot_synodic_covariance(filter_times, P_hist, moon_pos_filter, t_burn=t_burn)

    return gnss, filter_times, x_hist, P_hist


if __name__ == "__main__":
    main(animate=True)
    # main(animate=False)
