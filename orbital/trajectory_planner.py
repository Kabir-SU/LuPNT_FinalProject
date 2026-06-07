import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from abc import ABC, abstractmethod
from numpy.typing import NDArray

import AA278.project.misc.constants as constants

class PhaseType(Enum):
    """Enumeration of different phases of a trajectory"""
    DIRECT_IMPULSIVE_BURN = auto()
    CONTINUOUS_BURN = auto()
    COAST_TO_TIME = auto()
    COAST_TO_EVENT = auto()

class TrajectoryPhase(ABC):
    """Abstract class for represent the different phases in the trajectory"""
    name: str

    @property
    @abstractmethod
    def phase_type(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

class DirectImpulsiveBurn(TrajectoryPhase):
    """Child class representing an impulsive burn phase"""
    _name: str = "Impulsive Burn"

    def __init__(self, tof: float):
        self.tof = tof

    @property
    def phase_type(self) -> PhaseType:
        return PhaseType.DIRECT_IMPULSIVE_BURN

    @property
    def name(self) -> str:
        return self._name
    
    def apply(self, x, delta_v):
        """Apply the delta v to the state as a discontinuous update"""
        x_new = x
        x_new[3:6] += delta_v
        
        return x_new
    
class ProgradeLaw:
    """Purely prograde steering law for continuous burn"""
    def direction(self, t, x):
        v = x[3:6]
        return v / np.linalg.norm(v)
    
class ProgradeNormalLaw:
    """Steering law applying mostly prograde burn, with small normal burn to change inclination"""
    def __init__(self, alpha_deg):
        self.alpha_rad = np.deg2rad(alpha_deg)

    def direction(self, t, x):
        r = x[:3]
        v = x[3:6]
        h = np.cross(r, v)

        v_hat = v / np.linalg.norm(v)
        h_hat = h / np.linalg.norm(h)

        dir = np.cos(self.alpha_rad) * v_hat + np.sin(self.alpha_rad) * h_hat

        return dir / np.linalg.norm(dir)


    
class ContinuousBurn(TrajectoryPhase):
    """Child class representing a continuous burn plan"""
    _name: str = "Continuous Burn"

    def __init__(self,
                 burn_duration: float,
                 thrust_N: float,
                 isp_s: float,
                 direction_law):
        self.burn_duration = burn_duration
        self.thrust_N = thrust_N
        self.isp_s = isp_s
        self.direction_law = direction_law

    @property
    def phase_type(self) -> PhaseType:
        return PhaseType.CONTINUOUS_BURN

    @property
    def name(self) -> str:
        return self._name
    
class CoastToTime(TrajectoryPhase):
    """Child class representing a coast phase until a certain predetermined time"""
    _name: str = "Coast to Time"

    def __init__(self, t_final: float):
        self.t_final = t_final

    @property
    def phase_type(self) -> PhaseType:
        return PhaseType.COAST_TO_TIME
    
    @property
    def name(self) -> str:
        return self._name
    
class CoastToEvent(TrajectoryPhase):
    """Child class representing a coast phase until a certain event occurs"""
    _name: str = "Coast to Event"

    def __init__(self, t_max: float, events: list):
        self.t_max = t_max
        self.events = events

    @property
    def phase_type(self) -> PhaseType:
        return PhaseType.COAST_TO_EVENT
    
    @property
    def name(self) -> str:
        return self._name
    
@dataclass
class TrajectoryPlan:
    phases: list[TrajectoryPhase]
    
class TrajectoryPlanner:
    @abstractmethod
    def make_plan(self, x0, t0, tf) -> TrajectoryPlan:
        pass

class ImpulsiveTrajectoryPlan(TrajectoryPlanner):
    """Trajectory plan for an impulsive burn"""
    def __init__(self, t_burn, burn_duration):
        self.t_burn = t_burn
        self.burn_duration = burn_duration

    def make_plan(self, x0, t0, tf) -> TrajectoryPlan:
        return TrajectoryPlan(
            phases=[
                CoastToTime(t_final=self.t_burn),
                DirectImpulsiveBurn(self.burn_duration),
                CoastToEvent(t_max=tf, events=None)
            ]
        )
    
class ContinuousBurnTrajectoryPlan(TrajectoryPlanner):
    """Trajectory plan for a continuous burn trajectory with constant thrust"""
    def __init__(self,
                 burn_start_time: float,
                 burn_duration: float,
                 thrust_N: float,
                 isp_s: float,
                 direction_law):
        self.burn_start_time = burn_start_time
        self.burn_duration = burn_duration
        self.thrust_N = thrust_N
        self.isp_s = isp_s
        self.direction_law = direction_law

    def make_plan(self, x0, t0, tf):
        return TrajectoryPlan(
            phases=[
                CoastToTime(t_final=self.burn_start_time),
                ContinuousBurn(self.burn_duration, self.thrust_N, self.isp_s, self.direction_law),
                CoastToEvent(t_max=tf, events=None)
            ]
        )
