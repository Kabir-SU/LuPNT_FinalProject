# Important Events considered during trajectory

import numpy as np
import AA278.project.misc.constants as constants
import AA278.project.orbital.ephemeris as ephemeris

def earth_collision_event(t, x):
    """Event detecting collision with the earth's atmopsphere"""
    r_sat_eci = x[:3]
    true_alt = np.linalg.norm(r_sat_eci) - constants.R_EARTH

    return true_alt - constants.EARTH_MIN_ALT

earth_collision_event.terminal = True
earth_collision_event.direction = -1

def moon_collision_event(t, x, et0):
    """Event detecting collision with the moon (like lunar insertion)"""
    r_sat_eci = x[:3]
    r_moon_eci = ephemeris.get_lunar_pos(t + et0)
    r_moon_to_sat = r_sat_eci - r_moon_eci

    return np.linalg.norm(r_moon_to_sat) - (constants.R_MOON + 5000)

moon_collision_event.terminal = True
moon_collision_event.direction = 1

def moon_perilune_event(t, x, et0):
    """Event detecting collision with the moon (like lunar insertion)"""
    r_sat_eci = x[:3]
    r_moon_eci = ephemeris.get_lunar_pos(t + et0)
    r_moon_to_sat = r_sat_eci - r_moon_eci

    return np.linalg.norm(r_moon_to_sat) - (constants.R_MOON + 5000)

moon_perilune_event.terminal = True
moon_perilune_event.direction = -1

def dry_mass_event(t, x):
    return x[6] - 15.0

dry_mass_event.terminal = True
dry_mass_event.direction = -1

def blt_moon_apolune_event(t, x, et0):
    """Event used specifically for moon distance detection for BLT optimization"""
    r_sat_eci = x[:3]
    r_moon_eci = ephemeris.get_lunar_pos(t + et0)
    r_moon_to_sat = r_sat_eci - r_moon_eci

    return np.linalg.norm(r_moon_to_sat) - (constants.R_MOON + 10000)

blt_moon_apolune_event.terminal = True
blt_moon_apolune_event.direction = 1

def blt_moon_perilune_event(t, x, et0):
    """Event used specifically for moon distance detection for BLT optimization"""
    r_sat_eci = x[:3]
    r_moon_eci = ephemeris.get_lunar_pos(t + et0)
    r_moon_to_sat = r_sat_eci - r_moon_eci

    return np.linalg.norm(r_moon_to_sat) - (constants.R_MOON + 10000)

blt_moon_perilune_event.terminal = True
blt_moon_perilune_event.direction = -1