import random
from scipy.constants import pi, c
import numpy as np


INCLUDE_HORN_PHASE_SHIFT = True

INCLUDE_RANDOM_PHASE = False

# PARAMETERS

# Antenna parameters
Nx = 8  # number of elements in the x-direction
Ny = 12  # number of elements in the y-direction
dx = 0.5  # spacing between elements in the x-direction (in wavelengths)
dy = dx  # spacing between elements in the y-direction (in wavelengths)
freq_GHz = 27.5  # frequency (GHz)

system_losses_dB = 0
ep_max_gain_dBi = 0  # Max gain of the element pattern (EP)
cos_factor_theta = 1.2  # Raised cosine factor of the EP in theta
cos_factor_phi = 1.2  # Raised cosine factor of the EP in phi

# Derived antenna parameters
f = 1e9 * freq_GHz  # convert frequency to Hz
lambda_ = c / f  # wavelength (meters)
k = 2 * pi / lambda_  # wave vector

# Express grid spacing in meters
dx_m = dx * lambda_
dy_m = dy * lambda_

# Compute approximate aperture directivity
aperture_area = Nx * Ny * dx_m * dy_m
D = 4 * pi * aperture_area / lambda_ ** 2
D_dBi = 10 * np.log10(D)

# Estimate 3 dB beamwidth (BW) at broadside for the array aperature
beamwidth_broadside_x = 0.886 * lambda_ / (Nx * dx_m)
beamwidth_broadside_y = 0.886 * lambda_ / (Ny * dy_m)

# Number of Array Elements
num_elements = Nx * Ny

# Define element locations

# Element positions in x and y directions (assuming origin is at the center)
x = np.arange(Nx) - (Nx - 1) / 2
y = np.arange(Ny) - (Ny - 1) / 2

x = x * dx_m
y = y * dy_m

# Define mesh grid of element locations
X, Y = np.meshgrid(x, y)

# Transform X and Y into 1-D vectors
X_vec = X.reshape(-1)
Y_vec = Y.reshape(-1)

DIRECTED_POINTS = 1
directed_theta = np.array([30])
directed_phi = np.array([0])

theta0 = directed_theta[0] # Beam steering angle in theta (degrees)
phi0 = directed_phi[0] # 0Beam steering angle in phi (degrees)


# Define observation angles
theta_deg = np.linspace(-90, 90, 181)
phi_deg = np.linspace(-90, 90, 181)

theta = np.deg2rad(theta_deg)
phi = np.deg2rad(phi_deg)

# Make a meshgrid of theta and phi
THETA, PHI = np.meshgrid(theta, phi)








def azel_to_thetaphi(az, el):
    """ Az-El to Theta-Phi conversion.

    Args:
        az (float or np.array): Azimuth angle, in radians
        el (float or np.array): Elevation angle, in radians

    Returns:
      (theta, phi): Tuple of corresponding (theta, phi) angles, in radians
    """

    cos_theta = np.cos(el) * np.cos(az)
    # tan_phi = np.where(np.abs(np.sin(az)) < 1e-6, 0, np.tan(el) / np.sin(az)) # Avoid the divide by zero

    theta = np.arccos(cos_theta)
    phi = np.arctan2(np.tan(el), np.sin(az))
    phi = (phi + 2 * np.pi) % (2 * np.pi)

    return theta, phi


def thetaphi_to_azel(theta, phi):
    """ Az-El to Theta-Phi conversion.

    Args:
        theta (float or np.array): Theta angle, in radians
        phi (float or np.array): Phi angle, in radians

    Returns:
      (az, el): Tuple of corresponding (azimuth, elevation) angles, in radians
    """
    sin_el = np.sin(phi) * np.sin(theta)
    tan_az = np.cos(phi) * np.tan(theta)
    el = np.arcsin(sin_el)
    az = np.arctan(tan_az)

    return az, el
# Convert to azimuth and elevation
AZ, EL = thetaphi_to_azel(THETA, PHI)