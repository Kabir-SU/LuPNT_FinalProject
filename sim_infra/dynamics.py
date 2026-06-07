# Dynamics backbone for orbit propagation

import numpy as np
from scipy.integrate import solve_ivp
import spiceypy as spice

import AA278.project.misc.constants as constants
import AA278.project.orbital.ephemeris as ephemeris
import AA278.project.sim_infra.event as ev
import AA278.project.sim_infra.satellite as satellite
import AA278.project.misc.utils as utils

class Propagator:
    """Propagator object storing all data from propagation
    
    The propagation is done in the earth center inertial J2000 frame
    """

    def __init__(self, t0, dt, initial_state, sat: satellite.Satellite, et0, t_eval_step=1.0, max_step=60.0):
        """Constructor for propagator"""
        self._dt = dt
        self._t0 = t0
        self._initial_state = initial_state
        # state dimension
        # 3 positions, 3 velocities, mass
        self._n = 7
        self.et0 = et0
        # Only using physical characteristics of the satellite in question
        self._sat = sat
        self.t_eval_step = t_eval_step
        self.max_step = max_step

    def simulate(self, sim_duration, events, event_names, burn_plan=None):
        self.sun_pos_eci_time_hist = []
        self.moon_pos_eci_time_hist = []
        self.ephem_time = []

        def dynamics(t, x):
            et = self.et0 + (t - self._t0)
            lunar_pos = ephemeris.get_lunar_pos(et)
            sun_pos = ephemeris.get_sun_pos(et)

            # append ephem data to time history
            self.ephem_time.append(et)
            self.moon_pos_eci_time_hist.append(lunar_pos)
            self.sun_pos_eci_time_hist.append(sun_pos)

            a_grav = self._get_gravitation_acceleration(x[:3], r_moon=lunar_pos, r_sun=sun_pos)
            a_pert = self._get_perturbation_acceleration(x[:3], r_sun=sun_pos)
            
            a_thrust = 0.
            mdot = 0.
            if burn_plan:
                a_thrust, mdot = self._get_thrust_acceleration(t, x, burn_plan)
            
            xdot = x[3:6]
            vdot = a_grav + a_pert + a_thrust

            return [
                xdot[0],
                xdot[1],
                xdot[2],
                vdot[0],
                vdot[1],
                vdot[2],
                mdot,
            ]
        
        # start with the necessary checks of planet collision
        moon_event = lambda t, x: ev.moon_collision_event(t, x, self.et0)
        moon_event.terminal = True
        moon_event.direction = -1

        moon_event_p = lambda t, x: ev.moon_perilune_event(t, x, self.et0)
        moon_event_p.terminal = True
        moon_event_p.direction = 1
        all_events = [ev.earth_collision_event, moon_event, moon_event_p]
        all_event_names = ["Earth Collision", "Moon Collision", "Moon Collision"]
        
        # add mission planner specific events
        if events is not None and event_names is not None:
            for event, event_name in zip(events, event_names):
                all_events.append(event)
                all_event_names.append(event_name)

        # run the solver
        solution = solve_ivp(
            dynamics,
            (self._t0, self._t0 + sim_duration),
            self._initial_state,
            method="DOP853",
            rtol=1e-12,
            atol=[1e-9, 1e-9, 1e-9, 1e-12, 1e-12, 1e-12, 1e-9],
            max_step=self.max_step,
            t_eval=np.arange(self._t0, self._t0 + sim_duration + 1.0, self.t_eval_step),
            events=all_events,
        )

        # Determine termination criteria and return data to data storage 
        self.time_hist, self.state_time_hist = utils.post_process_integration(solution, all_event_names)

    def _get_thrust_acceleration(self, t, x, burn_plan):
        """Depending on the burn plan, get the acceleration due to thrust

        returns acceleration due to thrust and mass flow rate
        
        This is only used for continuous thrust profiles"""
        # handle the continuous burn
        thrust_unit_vector = burn_plan.direction_law.direction(t, x)

        mass_kg = x[6]
        a_thrust = (burn_plan.thrust_N / mass_kg) / 1000.0 * thrust_unit_vector
        m_dot = -burn_plan.thrust_N / (burn_plan.isp_s * constants.g0)

        return a_thrust, m_dot
        
        

    def _get_gravitation_acceleration(self, r_sat, r_moon, r_sun):
        """Get all graviational accelertion terms from sun, earth, moon

        r_sat: pos vector of satellite in ECI
        r_moon: pos vector of moon in ECI
        r_sun: pos vector of sun in ECI
        """
        # calculate the acceleration due to earth's gravity
        a_earth = -constants.MU_EARTH * r_sat / np.linalg.norm(r_sat)**3

        # calculate the acceleration due to the moon's gravity
        r_sat_to_moon = r_moon - r_sat
        r_sat_moon_norm = np.linalg.norm(r_sat_to_moon)
        r_earth_moon_norm = np.linalg.norm(r_moon)

        a_moon = constants.MU_MOON * (
            r_sat_to_moon / r_sat_moon_norm**3
            - r_moon / r_earth_moon_norm**3
        )

        # calculate the acceleration due to the sun's gravity
        r_sat_to_sun = r_sun - r_sat
        r_sat_sun_norm = np.linalg.norm(r_sat_to_sun)
        r_earth_sun_norm = np.linalg.norm(r_sun)

        a_sun = constants.MU_SUN * (
            r_sat_to_sun / r_sat_sun_norm**3
            - r_sun / r_earth_sun_norm**3
        )

        # add all the accelerations
        a_grav = a_earth + a_moon + a_sun
        
        return a_grav
    
    def _get_perturbation_acceleration(self, r_sat, r_sun):
        """Get all of the perturbation accelerations to be integrated
        
        Only considering J2 and SRP perturbation"""
        # get J2 perturbation acceleration
        a_j2 = self._get_j2_perturbation(r_sat)
        # get SRP perturbation acceleration
        a_srp = self._get_srp_perturbation(r_sat, r_sun)

        a_pert = a_j2 + a_srp

        return a_pert
    
    def _get_srp_perturbation(self, r_sat, r_sun):
        """Calculate Solar radiation pressure acceleration perturbation based upon sun ephemeris data"""
        # extract necessary variables
        P_SR = constants.P_SR
        AU = constants.AU_KM
        C_r = self._sat.cr
        A = self._sat.surf_area_m2
        m = self._sat.mass_kg

        # get vector from sun to sat and its norm
        r_sun_to_sat =  r_sat - r_sun
        r_sun_sat_mag = np.linalg.norm(r_sun_to_sat)

        # calculate SRP acceleration and convert to correct units
        a_srp_m_s2 = - P_SR * C_r * A / m * (AU / r_sun_to_sat**2) * r_sun_sat_mag
        a_srp_km_s2 = a_srp_m_s2 / 1000.0

        return a_srp_km_s2

    def _get_j2_perturbation(self, r_sat):
        """Calculate current J2 acceleration based upon satellite ECI position"""
        # extract necessary variables for j2 calculation
        x, y, z = r_sat
        r_mag = np.linalg.norm(r_sat)
        J2 = constants.J2
        mu = constants.MU_EARTH
        R_E = constants.R_EARTH

        # calculate the j2 perturbation based upon the standard equation
        coeff = 3 * J2 * mu * R_E**2 / (2 * r_mag**5)

        x_term = x * (5 * z**2 / r_mag**2 - 1)
        y_term = y * (5 * z**2 / r_mag**2 - 1)
        z_term = z * (5 * z**2 / r_mag**2 - 3)

        a = coeff * np.array([x_term, y_term, z_term])

        return a

    
    def get_results(self):
        """Returns time and state history for satellite"""
        return self.time_hist, self.state_time_hist
    
    def get_sun_ephemeris(self):
        """Return sun ephemeris time history"""
        return np.array(self.sun_pos_eci_time_hist)
    
    def get_moon_ephemeris(self):
        """Return moon ephemeris time history"""
        return np.array(self.moon_pos_eci_time_hist)
    
    def get_ephemeris_time(self):
        """Return ephemeris time """
        return np.array(self.ephem_time)
