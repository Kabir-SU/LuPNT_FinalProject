import numpy as np

import AA278.project.misc.constants as constants
import AA278.project.misc.utils as utils

class Satellite:
    """Satellite object storing its own state and burn plans"""

    def __init__(self, mass_kg=25., burn_type="Impulsive"):
        """Constructor of satellite"""
        # state dimension
        self._n = 7
        # State of x,y,z position and x,y,z velocities
        # These are cartesian pos/vel in the J2000 frame
        self._state = np.zeros(self._n)
        # Bool indicating the initial state isn't initialized
        self._initialized = False

        # set physical properties based upon CAPSTONE approximations
        self.mass_kg = mass_kg # kg
        self.surf_area_m2 = 2 * 0.34**2 + 4 * (0.34 * 0.64) # m^2
        self.cr = 1.5 # WAG for reflectivity coeff


    def init_state(self, pos, vel):
        """Set the initial position and velocity in ECI frame"""
        self._state[:3] = pos
        self._state[3:6] = vel
        self._state[6] = self.mass_kg
        self._initialized = True

    def initialize_from_coe(self, coe: utils.COE, mu):
        """Initialize the position and velocity from COEs"""
        self._state[:6] = utils.coe_to_cart(coe, mu)
        self._state[6] = self.mass_kg
        self._initialized = True

    def override_state(self, new_state):
        self._state = new_state

    def get_pos(self):
        """Return position in ECI frame"""
        return self._state[:3]

    def get_vel(self):
        """Return velocity in ECI frame"""
        return self._state[3:6]

    def get_state(self):
        return self._state

    def is_initialized(self):
        """Return whether initial state has been set or not"""
        return self._initialized
    
    def get_burn_config(self):
        """Return Burn configs"""
        return self.burn_config_idx

    def udpate_state(self, pos, vel):
        """Update the current state of the satellite (from propagator)"""
        self._state[:3] = pos
        self._state[3:6] = vel
