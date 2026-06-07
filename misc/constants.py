# Constants used throughout codebase for physics and mission planning

###########################
### Planetary constants ###
###########################
MU_MOON = 4902.800118 # km^3/s^2
MU_EARTH = 398600.442 # km^3/s^2
MU_SUN = 1.327e11 # km^3/s^2

R_EARTH = 6378.137 # km
R_MOON = 1737 # km
R_SUN = 696000 # km

# Earth J2 coefficient
J2 = 1.0826e-3
# SRP coefficients
P_SR = 4.56e-6 # N/m^2
# Astronomical Unit in km
AU_KM = 149597870.7 # km

########################
### UNIT CONVERSIONS ###
########################

# Time Unit Conversions
MIN_TO_SEC = 60
HOUR_TO_MIN = 60
DAY_TO_HOUR = 24
DAY_TO_SEC = DAY_TO_HOUR * HOUR_TO_MIN * MIN_TO_SEC

#######################
### EVENT DETECTION ###
#######################

# minimum earth altitude allowed
EARTH_MIN_ALT = 120 # km

g0 = 9.80665 # m/s^2
DT = 0.01 # sec

C_LIGHT = 299792.458 # km/s
