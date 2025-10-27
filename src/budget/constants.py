import numpy as np

# Numerical constants
epsilon = np.finfo(np.float64).eps

# Physical constants
g = 9.80665  # gravitational acceleration (m / s**2)
cp = 1004.0  # specific heat at constant pressure for dry air (J / kg / K)
ps = 1000e2  # reference surface pressure (Pa)
Rd = 287.058  # gas constant for dry air (J / kg / K)
chi = Rd / cp  # ~2/7 (dimensionless)
earth_radius = 6.3712e6  # mean Earth radius (m)

Omega = 2.0 * np.pi / (23 * 3600 + 56 * 60 + 4.1)  # Earth's rotation rate, (s**(-1))
