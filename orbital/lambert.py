import numpy as np
from scipy.optimize import fsolve

def lamberts_solver(r1, r2, time_of_flight, mu, short_transfer=True):
    """Given a starting and ending position vector and a time of transfer,
    it is possible to use lambert's equation to solve for the required velocity
    to complete the transfer from the first to last position vector.

    The method of doing this is using a universal variable formulation

    Returns:
    - v1: required velocity to reach final position vector in `time_of_flight` secs.
    - v2: velocity at the final position of the orbit arc
    """

    # get magnitudes of position vectors
    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    # calculate transfer angle
    dtheta = np.arccos(np.dot(r1, r2) / (r1_mag * r2_mag))

    if not short_transfer:
        dtheta = 2*np.pi - dtheta

    # define A (geometry parameter)
    A = np.sin(dtheta) * np.sqrt(r1_mag * r2_mag / (1 - np.cos(dtheta)))

    # define internal function used in time of flight equation
    def y_func(z):
        return r1_mag + r2_mag + A * (z * stumpff_s(z) - 1) / (np.sqrt(stumpff_c(z)))
    
    # define the time of flight equation, this depends on the desired time of flight along
    # with the variational parameter z
    def delta_time_of_flight_eq(z, desired_time_of_flight):
        y = y_func(z)
        C, S = stumpff_c(z), stumpff_s(z)

        return ((y/C)**(3/2) * S + A * np.sqrt(y)) / np.sqrt(mu) - desired_time_of_flight
    
    z = fsolve(delta_time_of_flight_eq, x0=0., args=(time_of_flight,))
    
    y = y_func(z)
    # calculate the required parameters for the universal variable formulation
    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_mag

    # solve for velocities
    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    return v1, v2


def stumpff_c(z):
    """Define the C stumpff function"""
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    else:
        return 0.5
    
def stumpff_s(z):
    """Define the S stumpff function"""
    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)**3)
    elif z < 0:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)**3)
    else:
        return 1/6
    
