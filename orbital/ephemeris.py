import spiceypy as spice
import numpy as np
import os

def load_kernels(data_dir):
    """Kernel loading (Set path to folder containing (naif0012.tls, de440s.bsp) in search_dirs)"""
    
    lsk = spk = None

    root = os.getcwd()
    d = os.path.join(root, os.path.normpath(data_dir))
    if lsk is None and os.path.isfile(os.path.join(d, "naif0012.tls")):
        lsk = os.path.join(d, "naif0012.tls")
    if spk is None and os.path.isfile(os.path.join(d, "de440s.bsp")):
        spk = os.path.join(d, "de440s.bsp")
    if lsk is None or spk is None:
        raise FileNotFoundError("SPICE kernels not found (naif0012.tls, de440.bsp).")
    spice.kclear()
    spice.furnsh(lsk)
    spice.furnsh(spk)
    print("Loaded SPICE Files!")

def get_lunar_pos(time):
    """Return Lunar position with respect to J2000 Earth frame"""
    lunar_pos, _ = spice.spkpos("MOON", time, "J2000", "NONE", "EARTH")
    return lunar_pos

def get_sun_pos(time):
    """Return sun position with respect to J2000 Earth frame"""
    sun_pos, _ = spice.spkpos("SUN", time, "J2000", "NONE", "EARTH")
    return sun_pos