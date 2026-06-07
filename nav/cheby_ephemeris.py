"""Chebyshev polynomial ephemeris for Moon and Sun - replaces per-step SPICE calls in the filter loop."""
import numpy as np
import spiceypy as spice


def _fit_one(body, observer, et0, etf, dt_sample, max_degree, tol):
    """
    Fit Chebyshev polynomials to a SPICE body position over [et0, etf].

    Returns coeffs (3, deg+1): coeffs[i] are the Chebyshev T coefficients
    for axis i, evaluated on the normalised interval t_bar in [-1, 1].
    """
    ets = np.linspace(et0, etf, max(int((etf - et0) / dt_sample) + 1, 2))
    states = np.array([spice.spkpos(body, et, "J2000", "NONE", observer)[0]
                       for et in ets])
    t_norm = 2.0 * (ets - et0) / (etf - et0) - 1.0

    coeffs = None
    rms = np.inf
    for deg in range(10, max_degree + 1):
        coeffs = np.array([
            np.polynomial.chebyshev.chebfit(t_norm, states[:, ax], deg)
            for ax in range(3)
        ])
        fitted = np.column_stack([
            np.polynomial.chebyshev.chebval(t_norm, coeffs[ax])
            for ax in range(3)
        ])
        rms = np.sqrt(np.mean(np.sum((fitted - states) ** 2, axis=1)))
        if rms < tol:
            break

    print(f"  Cheby {body}/{observer}: degree={coeffs.shape[1]-1}, RMS={rms:.3e} km")
    return coeffs


class ChebyEphemeris:
    """Fast Chebyshev polynomial ephemeris replacing SPICE calls for Moon and Sun."""

    def __init__(self, et0, etf, moon_coeffs, sun_coeffs):
        self._et0   = float(et0)
        self._scale = 2.0 / (float(etf) - float(et0))
        self._moon  = np.ascontiguousarray(moon_coeffs, dtype=np.float64)
        self._sun   = np.ascontiguousarray(sun_coeffs,  dtype=np.float64)

    def _t_bar(self, et):
        return self._scale * (et - self._et0) - 1.0

    def get_moon_pos(self, et):
        """Moon ECI position [km] relative to Earth, J2000 frame."""
        t = self._t_bar(et)
        return np.array([np.polynomial.chebyshev.chebval(t, self._moon[ax])
                         for ax in range(3)])

    def get_sun_pos(self, et):
        """Sun ECI position [km] relative to Earth, J2000 frame."""
        t = self._t_bar(et)
        return np.array([np.polynomial.chebyshev.chebval(t, self._sun[ax])
                         for ax in range(3)])

    @classmethod
    def build(cls, et0, etf, dt_sample=300.0, max_degree=30, tol=1e-4):
        """Fit Chebyshev polynomials for Moon and Sun over [et0, etf]."""
        print("Fitting Chebyshev ephemeris (one-time cost)...")
        moon_c = _fit_one("MOON", "EARTH", et0, etf, dt_sample, max_degree, tol)
        sun_c  = _fit_one("SUN",  "EARTH", et0, etf, dt_sample, max_degree, tol)
        print("Chebyshev ephemeris ready.")
        return cls(et0, etf, moon_c, sun_c)
