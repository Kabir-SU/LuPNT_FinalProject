from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

import AA278.project.misc.constants as constants
import AA278.project.sim_infra.dynamics as dynamics
import AA278.project.orbital.ephemeris as ephemeris
import AA278.project.orbital.lambert as lambert
import AA278.project.sim_infra.satellite as satellite
from AA278.project.orbital.trajectory_planner import TrajectoryPlan, TrajectoryPhase, PhaseType

from AA278.project.sim_infra.event import dry_mass_event

@dataclass
class MissionResult:
    t: NDArray[np.float]
    x: NDArray[np.float]
    sun_ephem: NDArray[np.float]
    moon_ephem: NDArray[np.float]
    ephem_time: NDArray[np.float]
    phase_names: list[str]

@dataclass
class PhaseResult:
    # time history of phase
    # for impulsive burn phase, this will be a singular time
    t: NDArray
    # state history of phase
    # for impulsive burn phase, this will one state vector
    x: NDArray
    # Sun ephemeris data
    # None for impulsive burn, since time doesn't advance
    sun_ephem: NDArray | None
    # Moon ephemeris data
    # None for impulsive burn, since time doesn't advance
    moon_ephem: NDArray | None
    # ephemeris time
    ephem_time: NDArray | None


class MissionPlanner:
    """Mission Planner for running simulation"""
    def __init__(self, sat: satellite.Satellite, et0):
        self.sat = sat
        self.et0 = et0
        self.et_cur = et0

    def run_phase(self, t0, x0, phase: TrajectoryPhase) -> PhaseResult:
        # match the phase of interest and propagate as applicable
        if phase.phase_type == PhaseType.DIRECT_IMPULSIVE_BURN:
             # define the propagator for this phase
            prop = dynamics.Propagator(t0=t0, dt=constants.DT, initial_state=x0, sat=self.sat, et0=self.et_cur)

            # get previous ephem time, use to calucalte moon final position
            et0 = self.et_cur
            et_moon = et0 + phase.tof
            final_moon_pos = ephemeris.get_lunar_pos(et_moon)

            # get desired delta v from lambert's solver
            v1, v2 = lambert.lamberts_solver(x0[:3], final_moon_pos, phase.tof, mu=constants.MU_EARTH)
            delta_v_vec = v1 - x0[3:6]

            print("Applied Delta V: ", delta_v_vec)
            print(np.linalg.norm(delta_v_vec), " km/s")

            # for an impulsive burn, we don't advance time at all
            # we only need to change the current velocity state discontinuously
            x_new = phase.apply(x0, delta_v_vec)

            return PhaseResult(
                t=np.array([t0]),
                x=np.array([x_new]),
                sun_ephem=None,
                moon_ephem=None,
                ephem_time=None,
            )
        
        if phase.phase_type == PhaseType.CONTINUOUS_BURN:
            # this is the logic for a continuous burn to the moon
            # initially, i will not consider targeting the moon, since that's
            # much more complex, this will just be a constant continuous thrust for now
            # define the propagator for this phase
            prop = dynamics.Propagator(t0=t0, dt=constants.DT, initial_state=x0, sat=self.sat, et0=self.et_cur)

            duration = phase.burn_duration

            events = [dry_mass_event]
            event_names = ["Out of Fuel"]
            prop.simulate(sim_duration=duration,
                          events=events,
                          event_names=event_names,
                          burn_plan=phase,
                          )
            t, x = prop.get_results()
            sun_ephem = prop.get_sun_ephemeris()
            moon_ephem = prop.get_moon_ephemeris()
            ephem_time = prop.get_ephemeris_time()

            return PhaseResult(
                t=t,
                x=x,
                sun_ephem=sun_ephem,
                moon_ephem=moon_ephem,
                ephem_time=ephem_time,
            )
            

        if phase.phase_type == PhaseType.COAST_TO_EVENT:
            # for coast to event, we need to run the propagator from the 
            # start to t_max with events in the loop
            # define the propagator for this phase
            prop = dynamics.Propagator(t0=t0, dt=constants.DT, initial_state=x0, sat=self.sat, et0=self.et_cur)

            duration = phase.t_max - t0
            events = []
            event_names = []
            prop.simulate(sim_duration=duration,
                          events=events,
                          event_names=event_names)
            t, x = prop.get_results()
            sun_ephem = prop.get_sun_ephemeris()
            moon_ephem = prop.get_moon_ephemeris()
            ephem_time = prop.get_ephemeris_time()

            return PhaseResult(
                t=t,
                x=x,
                sun_ephem=sun_ephem,
                moon_ephem=moon_ephem,
                ephem_time=ephem_time,
            )

        if phase.phase_type == PhaseType.COAST_TO_TIME:
            # for coast to time, we need to run the propagator from the start
            # to final time
            # define the propagator for this phase
            prop = dynamics.Propagator(t0=t0, dt=constants.DT, initial_state=x0, sat=self.sat, et0=self.et_cur)

            duration = phase.t_final - t0
            prop.simulate(sim_duration=duration, events=None, event_names=None)
            t, x = prop.get_results()
            sun_ephem = prop.get_sun_ephemeris()
            moon_ephem = prop.get_moon_ephemeris()
            ephem_time = prop.get_ephemeris_time()

            return PhaseResult(
                t=t,
                x=x,
                sun_ephem=sun_ephem,
                moon_ephem=moon_ephem,
                ephem_time=ephem_time,
            )

    def run_trajectory(self, t0, x0, traj_plan: TrajectoryPlan):
        phase_init_time = t0
        phase_init_state = x0

        time_hist = []
        state_hist = []
        sun_ephem_hist = []
        moon_ephem_hist = []
        ephem_time_hist = []
        phase_names = []
        for idx, phase in enumerate(traj_plan.phases):
            result = self.run_phase(phase_init_time, phase_init_state, phase)

            # set the initial state and time of the subsequent phase
            phase_init_state = result.x[:, -1]
            phase_init_time = result.t[-1]

            # add phase name
            phase_names.append(phase.name)
            print(phase.name)

            # if we are doing an impuslive burn, we shouldn't add to the time histories
            if phase.phase_type == PhaseType.DIRECT_IMPULSIVE_BURN:
                phase_init_state = result.x[-1]
                continue

            self.et_cur = result.ephem_time[-1]
            
            # add results to time history
            time_hist.append(result.t)
            state_hist.append(result.x)
            sun_ephem_hist.append(result.sun_ephem)
            moon_ephem_hist.append(result.moon_ephem)
            ephem_time_hist.append(result.ephem_time)
            

        return MissionResult(
            t=np.concatenate(time_hist),
            x=np.hstack(state_hist),
            sun_ephem=np.vstack(sun_ephem_hist),
            moon_ephem=np.vstack(moon_ephem_hist),
            ephem_time=np.concatenate(ephem_time_hist),
            phase_names=phase_names,
        )
