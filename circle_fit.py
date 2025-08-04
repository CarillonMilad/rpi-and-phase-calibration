import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def circle_residuals(params, x, y):
    """
    Residuals for circle fit.
    params = (cx, cy, r)
    """
    cx, cy, r = params
    return np.sqrt((x - cx)**2 + (y - cy)**2) - r

def fit_circle(x, y):
    """
    Fit a circle to points (x, y) using least squares.
    Returns center (cx, cy) and radius r.
    """
    # Initial guess: center = mean, radius = mean distance from mean
    cx0, cy0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - cx0)**2 + (y - cy0)**2))
    res = least_squares(circle_residuals, x0=(cx0, cy0, r0), args=(x, y))
    cx, cy, r = res.x
    return cx, cy, r

def extract_phase_from_AF_samples(AF_samples):

    # plt.plot(np.real(AF_samples), np.imag(AF_samples), 'o', label='Measured AF samples')
    # plt.show()

    """
    Given complex array factor samples for varying voltage on one element,
    fits a circle to the points, subtracts the center, and returns unwrapped phase.
    """
    x = np.real(AF_samples)
    y = np.imag(AF_samples)

    # Fit circle to measured points
    cx, cy, r = fit_circle(x, y)

    # Subtract center to isolate element phasor
    centered = AF_samples - (cx + 1j*cy)

    # Compute phase and unwrap
    phase = np.angle(centered)

    return phase, (cx, cy, r)

    # phase_unwrapped = np.unwrap(phase)

    # return phase_unwrapped, (cx, cy, r)


# --- Example usage ---
if __name__ == '__main__':
    # Simulate measurements for one element varying phase from 0 to 2pi
    N = 8  # total number of elements
    fixed_phasors = np.ones(N-1)  # others fixed at zero phase (1+0j)
    center = np.sum(fixed_phasors)  # center of circle in complex plane

    voltages = np.linspace(0, 100, 50)  # example voltage sweep
    element_phases = (voltages / 100) * 2 * np.pi  # simulate linear phase vs voltage

    AF_samples = np.array([center + np.exp(1j*phi) for phi in element_phases])

    # Extract phase from simulated measurements
    phase_unwrapped, (cx, cy, r) = extract_phase_from_AF_samples(AF_samples)

    print(f"Fitted circle center: ({cx:.3f}, {cy:.3f}), radius: {r:.3f}")

    # Plot results
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(np.real(AF_samples), np.imag(AF_samples), 'o', label='Measured AF samples')
    circle = plt.Circle((cx, cy), r, color='r', fill=False, linestyle='--', label='Fitted circle')
    plt.gca().add_patch(circle)
    plt.scatter(cx, cy, color='red', label='Circle center')
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.title('AF Samples and Fitted Circle')
    plt.axis('equal')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(voltages, phase_unwrapped, label='Extracted phase (unwrapped)')
    plt.xlabel('Voltage')
    plt.ylabel('Phase (radians)')
    plt.title('Element Phase vs Voltage')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
