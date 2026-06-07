"""
Optimal direct lunar transfer using minimum-delt-V launch window search.
"""

import warnings
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
import AA278.project.orbital.lambert as lambert
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

# Porkchop search

def compute_porkchop(
    et_window_start: float,
    r_dep: np.ndarray,
    v_dep: np.ndarray,
    dep_offsets_s: np.ndarray,
    tofs_s: np.ndarray,
    mu: float = MU_EARTH,
) -> np.ndarray:
    """
    Compute TLI delta V for every (departure epoch, ToF) grid cell.
    """
    N_dep = len(dep_offsets_s)
    N_tof = len(tofs_s)
    dv_matrix = np.full((N_dep, N_tof), np.nan)

    for i, dep_s in enumerate(dep_offsets_s):
        et_burn = et_window_start + dep_s    # epoch of the TLI burn
        for j, tof_s in enumerate(tofs_s):
            r_moon = ephemeris.get_lunar_pos(et_burn + tof_s)

            # Reject near-180 deg transfers - Lambert is degenerate there
            cos_angle = np.dot(r_dep, r_moon) / (np.linalg.norm(r_dep) * np.linalg.norm(r_moon))
            if cos_angle < -0.999:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    v1, _ = lambert.lamberts_solver(r_dep, r_moon, tof_s, mu=mu)
                dv = np.linalg.norm(v1 - v_dep)
                # Sanity check - unphysical solutions exceed ~10 km/s above LEO v
                if dv < 10.0:
                    dv_matrix[i, j] = dv
            except Exception:
                pass

    return dv_matrix


def plot_porkchop(
    dep_offsets_days: np.ndarray,
    tofs_days: np.ndarray,
    dv_matrix: np.ndarray,
    et_window_start: float,
    best_dep_day: float,
    best_tof_day: float,
    best_dv: float,
) -> tuple:
    """
    Contour plot: departure offset vs time-of-flight, coloured by DeltaV.
    """
    # Convert window-start epoch to a readable date string
    try:
        start_utc = spice.et2utc(et_window_start, "ISOC", 0)[:10]
    except Exception:
        start_utc = "window start"

    fig, ax = plt.subplots(figsize=(10, 7))

    # Contour levels in km/s
    dv_min = np.nanmin(dv_matrix)
    dv_max = np.nanpercentile(dv_matrix[np.isfinite(dv_matrix)], 99)

    print(dv_min)
    print(dv_max)
    print(dv_matrix)

    levels = np.linspace(dv_min, dv_max, 30)

    cf = ax.contourf(dep_offsets_days, tofs_days, dv_matrix.T,
                     levels=levels, cmap='viridis_r', extend='max')
    cs = ax.contour(dep_offsets_days, tofs_days, dv_matrix.T,
                    levels=levels[::4], colors='white', linewidths=0.5, alpha=0.6)
    ax.clabel(cs, fmt='%.2f km/s', fontsize=7, inline=True)

    plt.colorbar(cf, ax=ax, label='TLI DeltaV  [km/s]')

    # Mark the optimum
    ax.scatter([best_dep_day], [best_tof_day],
               marker='*', s=250, color='red', zorder=5,
               label=f'Optimum: DeltaV = {best_dv:.3f} km/s\n'
                     f'Dep +{best_dep_day:.2f} d, ToF {best_tof_day:.2f} d')
    ax.legend(fontsize=9, loc='upper right')

    ax.set_xlabel(f'Departure offset from {start_utc}  [days]')
    ax.set_ylabel('Time of flight  [days]')
    ax.set_title('Direct Lunar Transfer (28-day window)')
    ax.grid(True, linewidth=0.3, alpha=0.4)
    plt.tight_layout()
    plt.show()
    return fig, ax


# Performance plot

def _plot_nav_performance(times, states, filter_times, x_hist, P_hist, t_burn,
                          title="Pseudorange UDU - Optimal Transfer"):
    true_r = interp1d(times, states[:3],   axis=1, assume_sorted=True)(filter_times).T
    true_v = interp1d(times, states[3:6],  axis=1, assume_sorted=True)(filter_times).T

    pos_err = (x_hist[:, :3]  - true_r) * 1e3
    vel_err = (x_hist[:, 3:6] - true_v) * 1e3

    pos_rmse = np.sqrt(np.mean(pos_err**2, axis=0))
    vel_rmse = np.sqrt(np.mean(vel_err**2, axis=0))
    print(f"\n--- Navigation Filter RMSE ({title}) ---")
    print(f"  Position:  x={pos_rmse[0]:.2f} m,  y={pos_rmse[1]:.2f} m,  z={pos_rmse[2]:.2f} m")
    print(f"  Velocity: vx={vel_rmse[0]:.4f} m/s, vy={vel_rmse[1]:.4f} m/s, vz={vel_rmse[2]:.4f} m/s")
    print(f"  Total position RMSE : {np.sqrt(np.sum(pos_rmse**2)):.2f} m")
    print(f"  Total velocity RMSE : {np.sqrt(np.sum(vel_rmse**2)):.4f} m/s")

    pos_sig = np.sqrt(np.maximum(
        np.diagonal(P_hist[:, :3, :3],   axis1=1, axis2=2), 0)) * 1e3
    vel_sig = np.sqrt(np.maximum(
        np.diagonal(P_hist[:, 3:6, 3:6], axis1=1, axis2=2), 0)) * 1e3

    t_days     = filter_times / DAY_TO_SEC
    t_burn_day = t_burn / DAY_TO_SEC
    labels     = [('x', 'vx'), ('y', 'vy'), ('z', 'vz')]
    colors     = ['steelblue', 'seagreen', 'darkorange']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    fig.suptitle(title, fontsize=13)

    for i, (lp, lv) in enumerate(labels):
        ax = axes[0, i]
        ax.plot(t_days, pos_err[:, i], color=colors[i], linewidth=0.7, label='error')
        ax.fill_between(t_days, -3*pos_sig[:, i], 3*pos_sig[:, i],
                        alpha=0.25, color=colors[i], label='3sigma')
        ax.axhline(0, color='k', linewidth=0.4)
        ax.axvline(t_burn_day, color='red', linewidth=0.8, linestyle='--',
                   label='burn' if i == 0 else None)
        ax.set_ylabel(f'{lp} error  [m]')
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linewidth=0.4, alpha=0.5)

        ax = axes[1, i]
        ax.plot(t_days, vel_err[:, i], color=colors[i], linewidth=0.7, label='error')
        ax.fill_between(t_days, -3*vel_sig[:, i], 3*vel_sig[:, i],
                        alpha=0.25, color=colors[i], label='3sigma')
        ax.axhline(0, color='k', linewidth=0.4)
        ax.axvline(t_burn_day, color='red', linewidth=0.8, linestyle='--')
        ax.set_ylabel(f'{lv} error  [m/s]')
        ax.set_xlabel('Mission time  [days]')
        ax.set_ylim(-0.1, 0.1)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.show()
    return fig, axes

# Main function
def main(animate=False):
    # load ephemeris data!
    ephemeris.load_kernels(DATA_DIR)
    # initial parking orbit of 500 km altitude
    parking_orbit_alt_km = 500
    init_coe = COE(
        sma=R_EARTH + parking_orbit_alt_km,
        ecc=0.0,
        inc=np.deg2rad(0),
        raan=0.,
        arg_peri=0.,
        mean_anom=0.,
    )
    sat_ref = satellite.Satellite()
    sat_ref.initialize_from_coe(init_coe, MU_EARTH)
    r0 = sat_ref.get_pos()
    v0_leo = sat_ref.get_vel()

    orbit_period = 2 * np.pi * np.sqrt((R_EARTH + parking_orbit_alt_km)**3 / MU_EARTH)
    print(f"Parking orbit period: {orbit_period / MIN_TO_SEC:.1f} min")

    # 28-day window starting from the same reference as the other direct transfer scenario
    et_window_start = spice.str2et("2026-04-25T08:00:00")

    # Departure offsets: every 4 hours over 28 days
    dep_offsets_s = np.arange(0, 28 * DAY_TO_SEC + 1, 3600)
    dep_offsets_days = dep_offsets_s / DAY_TO_SEC

    # ToF: 2.5 to 7.0 days in 6-hour steps
    tofs_days = np.arange(2.5, 7.25, 0.25) 
    tofs_s =  tofs_days * DAY_TO_SEC

    print(f"Grid Search: {len(dep_offsets_s)} departure x {len(tofs_s)} ToF "
          f"= {len(dep_offsets_s) * len(tofs_s):,} Lambert solves")

    dv_matrix = compute_porkchop(
        et_window_start, r0, v0_leo,
        dep_offsets_s, tofs_s,
    )

    # Locate the global minimum for dv matrix
    flat_idx = np.nanargmin(dv_matrix)
    best_i, best_j = np.unravel_index(flat_idx, dv_matrix.shape)
    best_dep_day = dep_offsets_days[best_i]
    best_tof_day = tofs_days[best_j]
    best_dv_grid_search = dv_matrix[best_i, best_j]

    print(f"\nPorkchop optimum:")
    print(f"  Departure offset : {best_dep_day:.3f} days  "
          f"({best_dep_day * 24:.1f} h into the window)")
    print(f"  Time of flight   : {best_tof_day:.3f} days")
    print(f"  TLI DeltaV (search)  : {best_dv_grid_search*1e3:.1f} m/s")

    # Map departure offset to calendar date
    et_opt_burn = et_window_start + dep_offsets_s[best_i]
    try:
        burn_utc = spice.et2utc(et_opt_burn, "ISOC", 0)
        print(f"  Optimal burn epoch: {burn_utc}")
    except Exception:
        pass

    # plot_porkchop(
    #     dep_offsets_days, tofs_days, dv_matrix,
    #     et_window_start,
    #     best_dep_day, best_tof_day, best_dv_grid_search,
    # )

    # Run full trajectory at the optimal departure epoch
    # The simulation starts t_burn seconds before the TLI burn so the
    # spacecraft coasts briefly in the parking orbit before the Lambert solve.
    t_burn    = 15.5 * 60            # [s] coast before burn (same as direct transfer)
    tof_opt_s = best_tof_day * DAY_TO_SEC

    et0_sim = et_opt_burn - t_burn   # simulation start epoch

    num_days = t_burn / DAY_TO_SEC + best_tof_day - 0.15
    duration = num_days * DAY_TO_SEC
    print(f"\nRunning trajectory for {num_days:.3f} days "
          f"(coast {t_burn/60:.1f} min + TLI {best_tof_day:.2f} days)")

    sat = satellite.Satellite()
    sat.initialize_from_coe(init_coe, MU_EARTH)

    # The porkchop assumes departure from r0 at et_opt_burn. Back-propagate r0
    # along the circular LEO by t_burn so the spacecraft arrives at r0 exactly
    # at the burn epoch. For an equatorial circular orbit this is a z-axis rotation.
    omega_leo = np.sqrt(MU_EARTH / np.linalg.norm(r0)**3)
    theta     = omega_leo * t_burn
    cos_mt, sin_mt = np.cos(-theta), np.sin(-theta)
    r0_start = np.array([cos_mt*r0[0] - sin_mt*r0[1],
                         sin_mt*r0[0] + cos_mt*r0[1], r0[2]])
    v0_start = np.array([cos_mt*v0_leo[0] - sin_mt*v0_leo[1],
                         sin_mt*v0_leo[0] + cos_mt*v0_leo[1], v0_leo[2]])
    sat.init_state(r0_start, v0_start)
    initial_state = sat.get_state()

    trajectory_plan = ImpulsiveTrajectoryPlan(
        t_burn=t_burn,
        burn_duration=tof_opt_s,
    )
    phases = trajectory_plan.make_plan(x0=initial_state, t0=0, tf=duration)

    mission_sim = MissionPlanner(sat=sat, et0=et0_sim)
    results = mission_sim.run_trajectory(t0=0, x0=initial_state, traj_plan=phases)

    times  = results.t
    states = results.x
    moon_pos_time_hist = ephemeris.get_lunar_pos(et0_sim + times)

    # Report actual delta V from the mission planner
    burn_candidates = np.where(np.abs(times - t_burn) < 0.5)[0]
    if len(burn_candidates) >= 2:
        actual_dv = np.linalg.norm(
            states[3:6, burn_candidates[1]] - states[3:6, burn_candidates[0]]
        )
        print(f"  Actual mission-planner DeltaV: {actual_dv*1e3:.1f} m/s")

    # # Trajectory plot
    # if not animate:
    #     fig = plt.figure()
    #     ax  = fig.add_subplot(111, projection='3d')
    #     ax.plot(states[0], states[1], states[2], label='Orbit')
    #     ax.plot(moon_pos_time_hist[:, 0],
    #             moon_pos_time_hist[:, 1],
    #             moon_pos_time_hist[:, 2], label='Moon')
    #     utils.plot_earth(ax, R_EARTH=R_EARTH)
    #     ax.set_xlabel('x [km]'); ax.set_ylabel('y [km]'); ax.set_zlabel('z [km]')
    #     ax.legend(); utils.set_axes_equal(ax)
    #     ax.set_title(f'Optimal transfer: dep +{best_dep_day:.2f} d, '
    #                  f'ToF {best_tof_day:.2f} d, DeltaV {best_dv_grid_search*1e3:.0f} m/s')
    #     plt.show()
    # else:
    #     utils.animate_trajectory(
    #         states=states, time=times,
    #         moon_pos_time_hist=moon_pos_time_hist,
    #         R_EARTH=R_EARTH, interval=50, skip=1000, save_path="animation.gif",
    #     )

    # GNSS post-processing + navigation filter
    gnss = run_gnss_time_history(times, states, et0_sim, sample_rate_hz=1/10)
    # gnss.plot_overview()

    burn_dv_eci = np.zeros(3)
    if len(burn_candidates) >= 2:
        burn_dv_eci = states[3:6, burn_candidates[1]] - states[3:6, burn_candidates[0]]

    gamma_true = sat.cr * sat.surf_area_m2 / sat.mass_kg
    x0_true   = np.array([*states[:6, 0], 0.0, 0.0, gamma_true])
    sigma_init = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 0.1, 1e-6, gamma_true * 0.5])
    rng        = np.random.default_rng(42)
    x0_filter  = x0_true + rng.standard_normal(9) * sigma_init
    P0         = np.diag(sigma_init ** 2)

    print("Running pseudorange UDU filter")
    filter_times, x_hist, P_hist = run_pseudorange_filter(
        gnss=gnss, et0=et0_sim, x0=x0_filter, P0=P0,
        dynamics_fn=earth_ekf_dynamics,
        udu_filter_cls=UDUFilter,
        process_noise_fn=ekf_process_noise,
        sigma_acc=1e-7, clk_q1=CLK_Q1, clk_q2=CLK_Q2,
        burn_time=t_burn, burn_dv=burn_dv_eci, burn_sigma_dv=0.01,
    )
    print(f"Filter complete: {len(filter_times):,} epochs.")

    _plot_nav_performance(times, states, filter_times, x_hist, P_hist, t_burn)

    gnss.plot_nav_analysis()

    moon_pos_filter = ephemeris.get_lunar_pos(et0_sim + filter_times)
    utils.plot_synodic_covariance(filter_times, P_hist, moon_pos_filter, t_burn=t_burn)


if __name__ == '__main__':
    main(animate=True)
