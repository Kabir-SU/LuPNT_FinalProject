"""Batch GNSS measurement time history (L1 pseudorange, single-frequency)."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import AA278.project.misc.constants as constants
from AA278.project.misc.utils import get_rot_PQW_to_IJK
from AA278.project.sensors.gnss_constellation import GNSSConstellation
from AA278.project.sensors.gnss_sensor import SIGNAL_BANDS


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GNSSTimeHistory:
    """Full GNSS measurement time history. NaN = not tracked. Indices 0-23 GPS, 24-47 Galileo."""

    times_s:        np.ndarray
    rx_pos:         np.ndarray
    sv_pos:         np.ndarray
    sv_vel:         np.ndarray
    true_range:     np.ndarray
    pseudorange_L1: np.ndarray
    cnr_L1:         np.ndarray
    signal_type:    np.ndarray
    constellation:  GNSSConstellation

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_times(self) -> int:
        return len(self.times_s)

    @property
    def n_sats(self) -> int:
        return self.sv_pos.shape[1]

    # ------------------------------------------------------------------
    # Tracking status
    # ------------------------------------------------------------------

    def tracked_L1(self) -> np.ndarray:
        """(N_t, N_sats) bool - satellite tracked on L1 at this epoch."""
        return np.isfinite(self.pseudorange_L1)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_overview(self, day_axis: bool = True):
        """3-panel summary figure: altitude, visible satellite counts, mean C/N0."""
        t = self.times_s / constants.DAY_TO_SEC if day_axis else self.times_s
        xlabel = "Mission time  [days]" if day_axis else "Mission time  [s]"

        altitude = np.linalg.norm(self.rx_pos, axis=1) - constants.R_EARTH

        vis      = np.isfinite(self.cnr_L1)
        mainlobe = self.signal_type == 1
        sidelobe = self.signal_type == 2

        n_ml_gps = np.sum(vis[:, :24] & mainlobe[:, :24], axis=1)
        n_sl_gps = np.sum(vis[:, :24] & sidelobe[:, :24], axis=1)
        n_ml_gal = np.sum(vis[:, 24:] & mainlobe[:, 24:], axis=1)
        n_sl_gal = np.sum(vis[:, 24:] & sidelobe[:, 24:], axis=1)
        n_ml = n_ml_gps + n_ml_gal
        n_sl = n_sl_gps + n_sl_gal

        with np.errstate(all='ignore'):
            mean_cnr_L1 = np.nanmean(self.cnr_L1.astype(float), axis=1)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("GNSS Measurements", fontsize=13)

        axes[0].plot(t, altitude / 1e3, color='black', linewidth=0.8)
        axes[0].axhline(20.2, color='steelblue', linestyle='--', linewidth=0.8,
                        label='GPS altitude (~20 200 km)')
        axes[0].axhline(23.2, color='seagreen',  linestyle='--', linewidth=0.8,
                        label='Galileo altitude (~23 200 km)')
        axes[0].set_ylabel("Altitude  [x10^3 km]")
        axes[0].legend(fontsize=8, loc='upper left')
        axes[0].grid(True, linewidth=0.4, alpha=0.5)

        axes[1].stackplot(t, n_ml, n_sl,
                          labels=['Mainlobe (GPS+Galileo)', 'Sidelobe (GPS+Galileo)'],
                          colors=['steelblue', 'darkorange'], alpha=0.75)
        axes[1].plot(t, n_ml_gps, 'b--', linewidth=0.6, label='GPS mainlobe', alpha=0.9)
        axes[1].plot(t, n_ml_gal, 'g--', linewidth=0.6, label='Galileo mainlobe', alpha=0.9)
        axes[1].set_ylabel("Visible satellites")
        axes[1].set_ylim(bottom=0)
        axes[1].legend(fontsize=8, loc='upper right')
        axes[1].grid(True, linewidth=0.4, alpha=0.5)

        axes[2].plot(t, mean_cnr_L1, color='mediumpurple', linewidth=0.8,
                     label='L1 / E1  (1575 MHz)')
        axes[2].axhline(15.0, color='red', linestyle='--', linewidth=0.8,
                        label='Min threshold (15 dB-Hz)', alpha=0.8)
        axes[2].set_ylabel("Mean C/N0  [dB-Hz]")
        axes[2].set_xlabel(xlabel)
        axes[2].legend(fontsize=8)
        axes[2].grid(True, linewidth=0.4, alpha=0.5)

        plt.tight_layout()
        plt.show()
        return fig, axes

    def plot_nav_analysis(self, day_axis: bool = True, dop_threshold: float = 20.0):
        """
        3-panel navigation analysis: satellite visibility, GDOP/PDOP, and TDOP.

        Parameters
        ----------
        day_axis : bool
            If True the x-axis is in mission days, otherwise seconds.
        dop_threshold : float
            DOP values above this threshold are masked to NaN (shown as gaps).
            Prevents poor-geometry spikes (e.g. 1e6 in LEO) from hiding trends.
        """
        t = self.times_s / constants.DAY_TO_SEC if day_axis else self.times_s
        xlabel = "Mission time  [days]" if day_axis else "Mission time  [s]"

        tracked = self.tracked_L1()
        mainlobe = self.signal_type == 1
        sidelobe = self.signal_type == 2

        n_ml_gps = np.sum(tracked[:, :24] & mainlobe[:, :24], axis=1)
        n_ml_gal = np.sum(tracked[:, 24:] & mainlobe[:, 24:], axis=1)
        n_sl     = np.sum(tracked & sidelobe, axis=1)

        # --- DOP computation ---
        N_t   = self.n_times
        gdop  = np.full(N_t, np.nan)
        pdop  = np.full(N_t, np.nan)
        tdop  = np.full(N_t, np.nan)

        for k in range(N_t):
            vis_k = tracked[k]
            n_vis = int(vis_k.sum())
            if n_vis < 4:
                continue
            r_sv  = self.sv_pos[k, vis_k, :]
            e     = r_sv - self.rx_pos[k]
            e    /= np.linalg.norm(e, axis=1, keepdims=True)
            H     = np.column_stack([e, np.ones(n_vis)])
            HtH   = H.T @ H
            if np.linalg.matrix_rank(HtH) < 4:
                continue
            DOP     = np.linalg.inv(HtH)
            gdop[k] = np.sqrt(max(np.trace(DOP), 0.0))
            pdop[k] = np.sqrt(max(np.trace(DOP[:3, :3]), 0.0))
            tdop[k] = np.sqrt(max(DOP[3, 3], 0.0))

        gdop[gdop > dop_threshold] = np.nan
        pdop[pdop > dop_threshold] = np.nan
        tdop[tdop > dop_threshold] = np.nan

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Navigation Analysis - Satellite Geometry & DOP", fontsize=13)

        axes[0].stackplot(t, n_ml_gps, n_ml_gal,
                          labels=['GPS mainlobe', 'Galileo mainlobe'],
                          colors=['steelblue', 'seagreen'], alpha=0.75)
        axes[0].plot(t, n_sl, color='darkorange', linewidth=0.8,
                     linestyle='--', label='Sidelobe total', alpha=0.9)
        axes[0].set_ylabel("Satellites in view (L1)")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(fontsize=8, loc='upper right')
        axes[0].grid(True, linewidth=0.4, alpha=0.5)

        axes[1].plot(t, gdop, color='steelblue',  linewidth=0.8, label='GDOP')
        axes[1].plot(t, pdop, color='darkorange',  linewidth=0.8,
                     linestyle='--', label='PDOP')
        axes[1].axhline(6.0, color='red', linestyle=':', linewidth=0.8,
                        label='GDOP = 6 (marginal)', alpha=0.7)
        axes[1].set_ylabel("DOP")
        axes[1].set_ylim(bottom=0)
        axes[1].legend(fontsize=8, loc='upper right')
        axes[1].grid(True, linewidth=0.4, alpha=0.5)

        axes[2].plot(t, tdop, color='mediumpurple', linewidth=0.8, label='TDOP')
        axes[2].set_ylabel("TDOP")
        axes[2].set_ylim(bottom=0)
        axes[2].set_xlabel(xlabel)
        axes[2].legend(fontsize=8, loc='upper right')
        axes[2].grid(True, linewidth=0.4, alpha=0.5)

        plt.tight_layout()
        plt.show()
        return fig, axes


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_gnss_time_history(
    times: np.ndarray,
    states: np.ndarray,
    et0: float,
    sample_rate_hz: float = 1.0,
    include_sidelobe: bool = True,
    min_cnr_db_hz: float = 15.0,
    rx_gain_dBic: float = 0.0,
    noise_seed: int | None = None,
) -> GNSSTimeHistory:
    """Simulate an L1 GNSS receiver and return the full measurement time history."""
    rng = np.random.default_rng(noise_seed)

    # --- Time grid and trajectory interpolation ---
    dt_s   = 1.0 / sample_rate_hz
    t_gnss = np.arange(times[0], times[-1], dt_s)
    N_t    = len(t_gnss)
    print(f"  GNSS time history: {N_t:,} epochs at {sample_rate_hz:.3g} Hz ...")

    interp_pos = interp1d(times, states[:3], axis=1, assume_sorted=True)
    rx_pos = interp_pos(t_gnss).T   # (N_t, 3) km ECI

    # --- Constellation ---
    const  = GNSSConstellation(epoch_et=0.0)
    N_sats = len(const.satellites)   # 48

    R_list = [get_rot_PQW_to_IJK(s.inc, s.raan0, 0.0) for s in const.satellites]

    # --- Band constants ---
    bp_L1  = SIGNAL_BANDS['L1']
    LAM_L1 = constants.C_LIGHT / bp_L1['freq_hz']   # km
    ML_HALF = np.deg2rad(21.3)
    RE_SQ   = constants.R_EARTH ** 2
    N0      = -204.0   # dBW/Hz at 290 K

    # --- Pre-allocate output arrays ---
    sv_pos_out = np.empty((N_t, N_sats, 3), dtype=np.float64)
    sv_vel_out = np.empty((N_t, N_sats, 3), dtype=np.float32)
    true_range = np.full ((N_t, N_sats),    np.nan, dtype=np.float64)
    cnr_L1_out = np.full ((N_t, N_sats),    np.nan, dtype=np.float32)
    sig_type   = np.zeros((N_t, N_sats),    dtype=np.int8)
    vis_L1     = np.zeros((N_t, N_sats),    dtype=bool)

    # --- Pass 1: chunked vectorised geometry and link budget ---
    CHUNK = 86400
    for cs in range(0, N_t, CHUNK):
        ce = min(cs + CHUNK, N_t)
        nc = ce - cs

        rx_c = rx_pos[cs:ce]
        dt   = (et0 + t_gnss[cs:ce]) - const.epoch_et

        sv_p = np.empty((nc, N_sats, 3))
        sv_v = np.empty((nc, N_sats, 3))
        for j, (sat, R) in enumerate(zip(const.satellites, R_list)):
            u     = sat.m0 + sat.n * dt
            cos_u = np.cos(u)
            sin_u = np.sin(u)
            r_pqw = np.column_stack([ sat.a * cos_u,          sat.a * sin_u,         np.zeros(nc)])
            v_pqw = np.column_stack([-sat.a * sat.n * sin_u,  sat.a * sat.n * cos_u, np.zeros(nc)])
            sv_p[:, j, :] = r_pqw @ R.T
            sv_v[:, j, :] = v_pqw @ R.T

        sv_pos_out[cs:ce] = sv_p
        sv_vel_out[cs:ce] = sv_v.astype(np.float32)

        rx_b = rx_c[:, np.newaxis, :]

        d    = sv_p - rx_b
        d_sq = np.einsum('ijk,ijk->ij', d, d)

        dot_d_rx = np.einsum('ijk,ik->ij', d, rx_c)
        t_star   = np.clip(-dot_d_rx / d_sq, 0.0, 1.0)
        closest  = rx_b + t_star[:, :, np.newaxis] * d
        occluded = np.einsum('ijk,ijk->ij', closest, closest) < RE_SQ

        sv_r    = np.sqrt(np.einsum('ijk,ijk->ij', sv_p, sv_p))[:, :, np.newaxis]
        nadir   = -sv_p / sv_r
        to_rx   = -d
        rho     = np.sqrt(d_sq)
        to_rx_u = to_rx / rho[:, :, np.newaxis]
        cos_th  = np.clip(np.einsum('ijk,ijk->ij', nadir, to_rx_u), -1.0, 1.0)
        theta   = np.arccos(cos_th)

        td  = np.degrees(theta)
        G_t = np.select(
            [td < 21.3, td < 26.0, td < 40.0, td < 55.0, td < 75.0, td < 90.0],
            [13.0,       8.0,       3.0,       -3.0,       -8.0,      -13.0],
            default=-20.0,
        )

        fspl_L1  = 20.0 * np.log10(4.0 * np.pi * rho / LAM_L1)
        cnr_L1_c = bp_L1['p_tx_dBW'] + G_t + rx_gain_dBic - fspl_L1 - N0

        mainlobe_c = theta < ML_HALF
        vis_L1_c   = ~occluded & (cnr_L1_c >= min_cnr_db_hz)
        if not include_sidelobe:
            vis_L1_c &= mainlobe_c

        true_range[cs:ce] = np.where(vis_L1_c, rho, np.nan)
        cnr_L1_out[cs:ce] = np.where(vis_L1_c, cnr_L1_c, np.nan).astype(np.float32)
        sig_type[cs:ce]   = np.where(vis_L1_c,
                                     np.where(mainlobe_c, np.int8(1), np.int8(2)),
                                     np.int8(0)).astype(np.int8)
        vis_L1[cs:ce] = vis_L1_c

    # --- Pass 2: pseudorange noise ---
    cnr_L1_f = cnr_L1_out.astype(np.float64)
    sig_rho_L1 = np.where(
        vis_L1,
        bp_L1['sigma_ref_km'] * 10.0 ** ((bp_L1['cnr_ref_db_hz'] - cnr_L1_f) / 20.0),
        0.0,
    )

    pseudorange_L1 = np.where(
        vis_L1,
        true_range + rng.standard_normal((N_t, N_sats)) * sig_rho_L1,
        np.nan,
    )

    return GNSSTimeHistory(
        times_s=t_gnss,
        rx_pos=rx_pos,
        sv_pos=sv_pos_out,
        sv_vel=sv_vel_out,
        true_range=true_range,
        pseudorange_L1=pseudorange_L1,
        cnr_L1=cnr_L1_out,
        signal_type=sig_type,
        constellation=const,
    )
