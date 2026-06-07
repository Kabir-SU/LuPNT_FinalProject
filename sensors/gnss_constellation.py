"""
Analytical GNSS constellation propagator.

Nominal GPS (Walker 24/6/2) and Galileo (Walker 24/3/1) constellations
propagated as purely Keplerian circular orbits - no J2, no perturbations.
"""

import numpy as np
from dataclasses import dataclass

import AA278.project.misc.constants as constants
from AA278.project.misc.utils import get_rv_PQW, get_rot_PQW_to_IJK


@dataclass
class GNSSSatellite:
    sv_id: int           # unique 0-based index across both constellations
    constellation: str   # 'GPS' or 'Galileo'
    a: float             # semi-major axis [km]
    inc: float           # inclination [rad]
    raan0: float         # RAAN at epoch [rad]
    m0: float            # mean anomaly at epoch [rad]
    n: float             # mean motion [rad/s]


def _build_walker(
    constellation: str,
    a: float,
    inc_deg: float,
    n_planes: int,
    sats_per_plane: int,
    F: int,
    sv_id_offset: int,
) -> list[GNSSSatellite]:
    """
    Generate satellite list for a Walker T/P/F constellation (circular, e=0).

    Walker definition:
      T = n_planes * sats_per_plane  total satellites
      RAAN spacing = 360 / n_planes  degrees between planes
      In-plane spacing = 360 / sats_per_plane  degrees between slots
      Phase offset = F * 360 / T  degrees per plane
    """
    inc = np.deg2rad(inc_deg)
    n_motion = np.sqrt(constants.MU_EARTH / a**3)
    total_sats = n_planes * sats_per_plane
    sats = []
    for p in range(n_planes):
        raan = np.deg2rad(p * 360.0 / n_planes)
        for s in range(sats_per_plane):
            m0 = np.deg2rad(
                s * 360.0 / sats_per_plane
                + p * F * 360.0 / total_sats
            )
            sats.append(GNSSSatellite(
                sv_id=sv_id_offset + p * sats_per_plane + s,
                constellation=constellation,
                a=a,
                inc=inc,
                raan0=raan,
                m0=m0,
                n=n_motion,
            ))
    return sats


class GNSSConstellation:
    """
    Nominal GPS + Galileo constellation with analytical Keplerian propagation.

    Satellite ordering: indices 0-23 -> GPS, indices 24-47 -> Galileo.

    Parameters
    ----------
    epoch_et : float
        SPICE ephemeris time [s] at which the stored mean anomalies m0 are
        defined.  Default 0.0 corresponds to J2000.
    """

    # GPS Walker 24/6/2
    _GPS_A_KM = 26559.0
    _GPS_INC_DEG = 55.0
    _GPS_N_PLANES = 6
    _GPS_SATS_PER_PLANE = 4
    _GPS_F = 2

    # Galileo Walker 24/3/1
    _GAL_A_KM = 29600.0
    _GAL_INC_DEG = 56.0
    _GAL_N_PLANES = 3
    _GAL_SATS_PER_PLANE = 8
    _GAL_F = 1

    def __init__(self, epoch_et: float = 0.0):
        self.epoch_et = epoch_et
        gps_sats = _build_walker(
            'GPS', self._GPS_A_KM, self._GPS_INC_DEG,
            self._GPS_N_PLANES, self._GPS_SATS_PER_PLANE, self._GPS_F,
            sv_id_offset=0,
        )
        gal_sats = _build_walker(
            'Galileo', self._GAL_A_KM, self._GAL_INC_DEG,
            self._GAL_N_PLANES, self._GAL_SATS_PER_PLANE, self._GAL_F,
            sv_id_offset=len(gps_sats),
        )
        self.satellites: list[GNSSSatellite] = gps_sats + gal_sats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _propagate(self, sat: GNSSSatellite, et: float):
        """
        Returns (r_eci [km], v_eci [km/s]) for *sat* at ephemeris time *et*.

        For a circular orbit (e=0) the eccentric anomaly equals the argument
        of latitude: E = M0 + n*(t - t_epoch).
        """
        u = sat.m0 + sat.n * (et - self.epoch_et)   # argument of latitude [rad]
        R = get_rot_PQW_to_IJK(inc=sat.inc, raan=sat.raan0, omega=0.0)
        r_pqw, v_pqw = get_rv_PQW(a=sat.a, e=0.0, E=u, mu=constants.MU_EARTH)
        return R @ r_pqw, R @ v_pqw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_positions(self, et: float) -> np.ndarray:
        """
        ECI positions for all satellites at *et*.

        Returns
        -------
        ndarray, shape (N, 3)
            Positions in km.  Rows 0-23 = GPS, rows 24-47 = Galileo.
        """
        out = np.empty((len(self.satellites), 3))
        for i, sat in enumerate(self.satellites):
            out[i], _ = self._propagate(sat, et)
        return out

    def get_velocities(self, et: float) -> np.ndarray:
        """
        ECI velocities for all satellites at *et*.

        Returns
        -------
        ndarray, shape (N, 3)
            Velocities in km/s.
        """
        out = np.empty((len(self.satellites), 3))
        for i, sat in enumerate(self.satellites):
            _, out[i] = self._propagate(sat, et)
        return out

    def get_states(self, et: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Convenience method returning both positions and velocities in a single
        pass over the satellite list.

        Returns
        -------
        positions : ndarray (N, 3) [km]
        velocities : ndarray (N, 3) [km/s]
        """
        n = len(self.satellites)
        pos = np.empty((n, 3))
        vel = np.empty((n, 3))
        for i, sat in enumerate(self.satellites):
            pos[i], vel[i] = self._propagate(sat, et)
        return pos, vel
