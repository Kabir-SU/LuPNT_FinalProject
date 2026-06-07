"""GNSS link budget and pseudorange noise model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import AA278.project.misc.constants as constants
from AA278.project.sensors.gnss_constellation import GNSSConstellation, GNSSSatellite


# ---------------------------------------------------------------------------
# Signal band definitions
# ---------------------------------------------------------------------------

_N0_DBW_PER_HZ = -204.0   # thermal noise floor at 290 K [dBW/Hz]

# Mainlobe half-cone angle for nadir-pointing L-band phased array.
# Earth subtends ~13.9 deg / 12.4 deg from GPS / Galileo altitude.
_ML_HALF_ANGLE_DEG = 21.3

SIGNAL_BANDS: dict[str, dict] = {
    'L1': {
        'freq_hz': 1575.42e6,       # GPS L1 / Galileo E1 [Hz]
        'p_tx_dBW': 17.0,           # transmit power ~50 W [dBW]
        'sigma_ref_km': 0.001,      # 1 m pseudorange noise at cnr_ref
        'cnr_ref_db_hz': 45.0,
    },
}


# ---------------------------------------------------------------------------
# Antenna gain model
# ---------------------------------------------------------------------------

def _antenna_gain_dBic(theta_rad: float) -> float:
    """Piecewise GPS/Galileo nadir transmit antenna gain [dBic] vs off-boresight angle [rad]."""
    theta_deg = math.degrees(theta_rad)
    if theta_deg < 21.3:
        return 13.0
    elif theta_deg < 26.0:
        return 8.0
    elif theta_deg < 40.0:
        return 3.0
    elif theta_deg < 55.0:
        return -3.0
    elif theta_deg < 75.0:
        return -8.0
    elif theta_deg < 90.0:
        return -13.0
    else:
        return -20.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _earth_occluded(r_rx: np.ndarray, r_sv: np.ndarray) -> bool:
    """Return True if Earth occludes the line of sight from r_rx to r_sv (both km ECI)."""
    d = r_sv - r_rx
    d_sq = float(np.dot(d, d))
    # t that minimises |r_rx + t*d|^2
    t = float(-np.dot(d, r_rx)) / d_sq
    t = max(0.0, min(1.0, t))
    closest = r_rx + t * d
    return float(np.dot(closest, closest)) < constants.R_EARTH ** 2


def _off_boresight_angle(r_rx: np.ndarray, r_sv: np.ndarray) -> float:
    """Off-boresight angle [rad] between satellite nadir and direction to receiver."""
    nadir = -r_sv / np.linalg.norm(r_sv)
    to_rx = r_rx - r_sv
    to_rx = to_rx / np.linalg.norm(to_rx)
    cos_theta = float(np.clip(np.dot(nadir, to_rx), -1.0, 1.0))
    return math.acos(cos_theta)


# ---------------------------------------------------------------------------
# Link budget
# ---------------------------------------------------------------------------

def _received_power_dBW(
    r_rx: np.ndarray,
    r_sv: np.ndarray,
    theta_rad: float,
    band: str,
    rx_gain_dBic: float,
) -> float:
    """Received signal power [dBW] from the free-space link budget."""
    bp = SIGNAL_BANDS[band]
    lam_km = constants.C_LIGHT / bp['freq_hz']
    rho_km = float(np.linalg.norm(r_rx - r_sv))

    G_t = _antenna_gain_dBic(theta_rad)
    fspl = 20.0 * math.log10(4.0 * math.pi * rho_km / lam_km)
    return bp['p_tx_dBW'] + G_t + rx_gain_dBic - fspl


def _cnr_db_hz(P_r_dBW: float) -> float:
    """Convert received power [dBW] to C/N0 [dB-Hz] assuming 290 K noise."""
    return P_r_dBW - _N0_DBW_PER_HZ


def _pseudorange_sigma_km(cnr: float, band: str) -> float:
    """Pseudorange noise std-dev [km] scaled from sigma_ref by C/N0."""
    bp = SIGNAL_BANDS[band]
    return bp['sigma_ref_km'] * 10.0 ** ((bp['cnr_ref_db_hz'] - cnr) / 20.0)


# ---------------------------------------------------------------------------
# Measurement dataclass
# ---------------------------------------------------------------------------

@dataclass
class GNSSMeasurement:
    sv_id: int
    constellation: str           # 'GPS' or 'Galileo'
    signal_band: str             # 'L1'
    signal_type: str             # 'mainlobe' or 'sidelobe'
    sv_pos_eci: np.ndarray       # satellite ECI position [km]
    sv_vel_eci: np.ndarray       # satellite ECI velocity [km/s]
    true_range: float            # geometric range [km]
    pseudorange: float           # noisy pseudorange [km]
    range_rate: float            # true range-rate [km/s]; NaN if v_rx=None
    range_rate_meas: float       # noisy range-rate [km/s]; NaN if v_rx=None
    off_boresight_angle: float   # transmit off-boresight angle [rad]
    cnr_db_hz: float             # carrier-to-noise density [dB-Hz]


# ---------------------------------------------------------------------------
# Sensor class
# ---------------------------------------------------------------------------

class GNSSSensor:
    """GNSS pseudorange and Doppler measurement generator."""

    def __init__(
        self,
        constellation: GNSSConstellation,
        bands: list[str] | None = None,
        include_sidelobe: bool = True,
        min_cnr_db_hz: float = 15.0,
        rx_gain_dBic: float = 0.0,
        noise_seed: int | None = None,
    ):
        self.constellation = constellation
        self.bands = bands if bands is not None else list(SIGNAL_BANDS.keys())
        for b in self.bands:
            if b not in SIGNAL_BANDS:
                raise ValueError(f"Unknown band '{b}'. Valid: {list(SIGNAL_BANDS)}")
        self.include_sidelobe = include_sidelobe
        self.min_cnr_db_hz = min_cnr_db_hz
        self.rx_gain_dBic = rx_gain_dBic
        self._rng = np.random.default_rng(noise_seed)

    def get_measurements(
        self,
        r_rx: np.ndarray,
        et: float,
        v_rx: np.ndarray | None = None,
    ) -> list[GNSSMeasurement]:
        """Return GNSSMeasurement list for all visible satellites at the given epoch."""
        sv_pos, sv_vel = self.constellation.get_states(et)
        measurements: list[GNSSMeasurement] = []
        ml_half_rad = math.radians(_ML_HALF_ANGLE_DEG)

        for i, sat in enumerate(self.constellation.satellites):
            r_sv = sv_pos[i]
            v_sv = sv_vel[i]

            # --- Geometry: band-independent ---
            if _earth_occluded(r_rx, r_sv):
                continue

            theta = _off_boresight_angle(r_rx, r_sv)
            is_mainlobe = theta < ml_half_rad
            signal_type = 'mainlobe' if is_mainlobe else 'sidelobe'
            if not is_mainlobe and not self.include_sidelobe:
                continue

            true_range = float(np.linalg.norm(r_rx - r_sv))
            if v_rx is not None:
                los = (r_rx - r_sv) / true_range
                range_rate = float(np.dot(los, v_rx - v_sv))
            else:
                range_rate = float('nan')

            # --- Per-band link budget + noise ---
            for band in self.bands:
                P_r = _received_power_dBW(r_rx, r_sv, theta, band, self.rx_gain_dBic)
                cnr = _cnr_db_hz(P_r)
                if cnr < self.min_cnr_db_hz:
                    continue

                sigma_rho = _pseudorange_sigma_km(cnr, band)
                pseudorange = true_range + float(self._rng.normal(0.0, sigma_rho))

                if v_rx is not None:
                    sigma_dot = sigma_rho * 0.01
                    range_rate_meas = range_rate + float(
                        self._rng.normal(0.0, sigma_dot)
                    )
                else:
                    range_rate_meas = float('nan')

                measurements.append(GNSSMeasurement(
                    sv_id=sat.sv_id,
                    constellation=sat.constellation,
                    signal_band=band,
                    signal_type=signal_type,
                    sv_pos_eci=r_sv.copy(),
                    sv_vel_eci=v_sv.copy(),
                    true_range=true_range,
                    pseudorange=pseudorange,
                    range_rate=range_rate,
                    range_rate_meas=range_rate_meas,
                    off_boresight_angle=theta,
                    cnr_db_hz=cnr,
                ))

        return measurements
