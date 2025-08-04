import random
import matplotlib.pyplot as plt
import horn_shifts
from vars import *


def print_antenna_parameters():
    # Print derived antenna parameters
    print('Estimated aperture directivity (dBi):', np.round(D_dBi, 1))
    print('Estimated 3 dB beamwidth at broadside (x):', np.round(beamwidth_broadside_x * 180 / pi, 1), 'degrees')
    print('Estimated 3 dB beamwidth at broadside (y):', np.round(beamwidth_broadside_y * 180 / pi, 1), 'degrees')
    print('Aperture dimensions (x):', np.round(Nx * dx_m, 2), 'meters')
    print('Aperture dimensions (y):', np.round(Ny * dy_m, 2), 'meters')
    print('Apreture area:', np.round(aperture_area, 2), 'm^2')
    print('Total number of antenna elements:', num_elements)


def plot_element_locations():
    # Plot the element locations X and Y using a scatter plot.
    plt.figure()
    plt.scatter(X, Y)
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title('Phased Array Element Locations')
    plt.grid(True)
    plt.axis('equal')
    plt.show()





def steering_vector(k, xv, yv, theta_deg, phi_deg):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    kx = k * np.sin(theta) * np.cos(phi)
    ky = k * np.sin(theta) * np.sin(phi)

    # Calculate the phase shift for each element
    phase_weights = np.exp(1j * (kx * xv + ky * yv))

    return phase_weights



def plot_phase_shifts(phase_shift_deg):
    plt.figure()
    plt.imshow(phase_shift_deg, clim=(-180, 180))
    plt.colorbar(label='Phase Shift (deg)')
    plt.title('Elemental Phase Shifts for Beam Steering Angle\nTheta = {}°, Phi = {}°'.format(theta0, phi0))
    plt.xlabel('X (idx)')
    plt.ylabel('Y (idx)')
    plt.show()


def dBi_to_linear(dBi):
    """Converts dBi to linear scale."""
    return 10 ** (dBi / 10)


def antenna_element_pattern(theta: np.ndarray, phi: np.ndarray,
                            cos_factor_theta: float = 1.0, cos_factor_phi: float = 1.0,
                            max_gain_dBi: float = 0.0) -> np.ndarray:
    """
    Calculates the radiation pattern of a single antenna element using a raised cosine model.

    Args:
      theta: Elevation angles in radians (numpy.ndarray).
      phi: Azimuth angles in radians (numpy.ndarray).
      cos_factor_theta: Cosine power factor for theta (float, default=1.0).
      cos_factor_phi: Cosine power factor for phi (float, default=1.0).
      max_gain_dBi: Maximum gain of the element pattern in dBi (float, default=0.0).

    Returns:
      A numpy array containing the element pattern values in linear scale.
    """

    # Convert max gain from dBi to linear scale
    max_gain = dBi_to_linear(max_gain_dBi)

    # Calculate the radiation pattern
    pattern = max_gain * np.cos(theta) ** cos_factor_theta * np.cos(phi) ** cos_factor_phi
    return pattern


def test_antenna_element_pattern():
    theta = np.radians(np.linspace(-90, 90, 180))
    phi = np.radians(np.linspace(-90, 90, 180))

    THETA, PHI = np.meshgrid(theta, phi)
    pattern = antenna_element_pattern(THETA, PHI)

    print(pattern.shape)

    # Plot the radiation pattern in the elevation plane (phi = 0)
    plt.figure()
    plt.polar(phi, pattern[90], 'b-')
    plt.title('Element Radiation Pattern in the Elevation Plane (Phi = 0°)')
    plt.ylabel('Normalized Gain')
    plt.xlabel('Azimuth Angle (degrees)')
    plt.show()

    # Plot the radiation pattern in the azimuth plane (theta = 0)
    plt.figure()
    plt.polar(theta, pattern[90], 'r-')
    plt.title('Element Radiation Pattern in the Azimuth Plane (Theta = 0°)')
    plt.ylabel('Normalized Gain')
    plt.xlabel('Elevation Angle (degrees)')
    plt.show()


# print(THETA, PHI)


# WITHOUT WEIGHTS INCLUDED
def AF(theta, phi, x, y, w, k):
    """
    Calculates the array factor for a given set of angles, coordinates, weights, and wave number.

    Args:
      theta: Elevation angle in radians.
      phi: Azimuth angle in radians.
      x: X-coordinates of the antenna elements.
      y: Y-coordinates of the antenna elements.
      w: Complex weights of the antenna elements.
      k: Wave number.

    Returns:
      The array factor as a complex number.
    """

    # w_vec = w.reshape(-1)
    # AF = np.sum(w_vec)
    # return AF

    N = len(x)  # Number of antenna elements

    # Calculate the phase shift for each antenna element
    phase_shift = -1j * k * (x * np.sin(theta) * np.cos(phi) + y * np.sin(theta) * np.sin(phi))

    # Reshape the complex weights into a 1-D vector
    w_vec = w.reshape(-1)


    # Multiply the weights by the phase shift and sum them up
    AF = np.sum(w_vec * np.exp(phase_shift))
    # print()
    # print(w_vec)
    # print(np.exp(phase_shift))
    # print()
    return AF



def plot_single_element_pattern(element_pattern):
    print('Element Pattern Shape:', element_pattern.shape)

    # Make an interpolated scatter plot using THETA, PHI, and element_pattern with color shading based on element_pattern magnitude

    # Create the scatter plot
    plt.figure()
    plt.scatter(np.degrees(THETA), np.degrees(PHI), c=abs(abs(element_pattern)), cmap='viridis')
    plt.colorbar()

    # Add labels and title
    plt.xlabel('THETA (deg)')
    plt.ylabel('PHI (deg)')
    plt.title('Average Single Element Pattern')
    plt.xlim([-90, 90])
    plt.ylim([-90, 90])

    # Show the plot
    plt.show()


def wrap_angle(angle):
    """
    Wraps an angle value between 0 and 2 pi.

    Args:
      angle: The angle value in radians.

    Returns:
      The wrapped angle value between 0 and 2 pi.
    """

    return angle % (2 * np.pi)


def test_wrap_angle():
    angle_in_radians = 10  # Example angle
    wrapped_angle = wrap_angle(angle_in_radians)
    print(f"Wrapped angle: {np.round(wrapped_angle, 2)} radians")


# phase_weights = steering_vector(k=k,
#                                 xv=X,
#                                 yv=Y,
#                                 theta_deg=theta0,
#                                 phi_deg=phi0)



# phase_shift_rad = np.angle(phase_weights)

# phase_shift_deg = np.degrees(phase_shift_rad)

# print(phase_shift_deg.shape)
# print(phase_shift_deg)

def phase_shift_deg_to_weight(phase_shift_deg):
    return np.exp(1j * phase_shift_deg / 180.0 * pi)

def add_horn_phase(phase_weights):
    pw = phase_weights * horn_shifts.HORN_PHASE_WEIGHT
    return pw

random_phase = np.zeros((Ny, Nx), dtype=complex)
for i in range(Ny):
    for j in range(Nx):
        random_phase[i][j] = phase_shift_deg_to_weight(random.uniform(-180, 180))

def add_random_phase(phase_weights):
    pw = phase_weights * random_phase
    return pw

def get_gain_plot(phase_weights):

    # Compute element pattern over all THETA, PHI angles
    element_pattern = antenna_element_pattern(THETA, PHI,
                                              cos_factor_theta,
                                              cos_factor_phi,
                                              max_gain_dBi=ep_max_gain_dBi)

    if (INCLUDE_HORN_PHASE_SHIFT):
        phase_weights = add_horn_phase(phase_weights)
    if (INCLUDE_RANDOM_PHASE):
        phase_weights = add_random_phase(phase_weights)

    # Calculate the array factor for each angle
    array_factor = np.zeros((len(theta), len(phi)), dtype=complex)
    for i, thi in enumerate(theta):
        for j, phj in enumerate(phi):
            # array_factor[i, j] = 1 * AF(thi, phj, x=X_vec, y=Y_vec, w=phase_weights, k=k)
            array_factor[i, j] = element_pattern[i, j] * AF(thi, phj, x=X_vec, y=Y_vec, w=phase_weights, k=k)

    array_factor_dB = 10 * np.log10(abs(array_factor))

    # Normalize array_factor_dB
    array_factor_dB_norm = array_factor_dB - np.max(array_factor_dB)

    # Normalize array_factor_dB
    # array_gain_dBi = D_dBi - system_losses_dB + array_factor_dB_norm
    array_gain_dBi = array_factor_dB_norm

    # If a value in power_pattern is less than a minimum threshold, set it to the minimum treshold for visualization purposes
    min_thres = np.max(array_factor_dB) - 50
    array_factor_plot_dB = array_factor_dB.clip(min=min_thres)

    min_thres = np.max(array_gain_dBi) - 50
    array_gain_plot_dBi = array_gain_dBi.clip(min=min_thres)

    return array_gain_plot_dBi


# TODO
def plot_gain_fft(phase_weights):

    near_field_data = np.exp(-((X ** 2 + Y ** 2) / (0.01 ** 2))) * phase_weights

    # pad
    nfd = np.zeros((181, 181))
    for i in range(86, 93 + 1):
        for j in range(86, 93 + 1):
            nfd[i, j] = near_field_data[i - 86, j - 86]


    # 2. Apply 2D FFT
    far_field_spectrum = np.fft.fftshift(np.fft.fft2(nfd))

    # 3. Calculate Far-Field Pattern (e.g., magnitude)
    far_field_magnitude = np.abs(far_field_spectrum)

    #gain_plot = 2*pi*far_field_magnitude

    plt.imshow(20 * np.log10(far_field_magnitude), origin='lower', extent=[x.min(), x.max(), y.min(), y.max()])
    #plt.imshow(gain_plot, origin='lower')
    plt.colorbar(label='Magnitude (dB)')
    plt.title('Antenna Far-Field Pattern (Magnitude)')
    plt.xlabel('Spatial Frequency (related to angle)')
    plt.ylabel('Spatial Frequency (related to angle)')
    plt.show()


def plot_pattern(phase_weights):
    array_gain_plot_dBi = get_gain_plot(phase_weights)
    # array_gain_plot_dBi = gain_fft(phase_weights)



    # Create the scatter plot
    plt.figure()
    plt.scatter(np.degrees(THETA), np.degrees(PHI), c=array_gain_plot_dBi.T, cmap='viridis')
    plt.colorbar(label='Gain (dBi)')

    # Add labels and title
    plt.title(f'Radiation Pattern Gain: Beam Steered to Theta = {theta0}°, Phi = {phi0}°')
    plt.xlabel('THETA (deg)')
    plt.ylabel('PHI (deg)')
    plt.xlim([np.min(theta_deg), np.max(theta_deg)])
    plt.ylim([np.min(phi_deg), np.max(phi_deg)])
    plt.xticks(np.arange(phi_deg[0], phi_deg[-1] + 1, 30))
    plt.yticks(np.arange(theta_deg[0], theta_deg[-1] + 1, 30))

    # Show the plot
    plt.show()


def plot_stuff(stuff):
    plt.figure()
    plt.scatter(np.degrees(THETA), np.degrees(PHI), c=stuff.T, cmap='viridis')
    plt.colorbar(label='Gain (dBi)')

    # Add labels and title
    plt.title(f'Radiation Pattern Gain: Beam Steered to Theta = {theta0}°, Phi = {phi0}°')
    plt.xlabel('THETA (deg)')
    plt.ylabel('PHI (deg)')
    plt.xlim([np.min(theta_deg), np.max(theta_deg)])
    plt.ylim([np.min(phi_deg), np.max(phi_deg)])
    plt.xticks(np.arange(phi_deg[0], phi_deg[-1] + 1, 30))
    plt.yticks(np.arange(theta_deg[0], theta_deg[-1] + 1, 30))

if __name__ == '__main__':

    phase_weights_correct = steering_vector(k=k,
                                    xv=X,
                                    yv=Y,
                                    theta_deg=theta0,
                                    phi_deg=phi0)
    phase_shift_rad = np.angle(phase_weights_correct)
    phase_shift_deg = np.degrees(phase_shift_rad)
    print(phase_shift_deg)
    plot_gain_fft(phase_weights_correct)
    plot_pattern(phase_weights_correct)