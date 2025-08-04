from vars import *
import numpy as np
import matplotlib.pyplot as plt
import math

Z_const = 6.875*2.54/100
horn_theta_t = np.radians(30)
f = freq_GHz*1e9


# Define new tilted coordinate system for the spherical wavefront
# Assume the source is tilted by theta_t about the y-axis
# Compute new radial distance considering tilt
Phi_spherical_tilted = np.zeros([Ny, Nx])

for i in range(0, Ny):
    for j in range(0, Nx):
        R_tilted = np.sqrt(np.power((x[j] - Z_const * np.sin(horn_theta_t)), 2) + np.power(y[i], 2) + np.power((Z_const * np.cos(horn_theta_t)), 2))
        # Compute phase for the tilted spherical wavefront
        Phi_spherical_tilted[i, j] = k * R_tilted

        # print(R_tilted)
HORN_PHASE_SHIFT = Phi_spherical_tilted
HORN_PHASE_SHIFT = np.mod(Phi_spherical_tilted, 2 * np.pi)
HORN_PHASE_SHIFT = HORN_PHASE_SHIFT * 180/ np.pi

def phase_deg_to_weight(phase_shift_deg):
    return np.exp(1j * phase_shift_deg / 180.0 * pi)

def phase_rad_to_weight(phase_shift_rad):
    return np.exp(1j * phase_shift_rad)

HORN_PHASE_WEIGHT = phase_deg_to_weight(HORN_PHASE_SHIFT)


def horn_phase_shift(x_loc, y_loc):
    r_tilted = np.sqrt(np.power((x_loc - Z_const * np.sin(horn_theta_t)), 2) + np.power(y_loc, 2) + np.power(
        (Z_const * np.cos(horn_theta_t)), 2))
    phi_spherical_tilted = k * r_tilted
    phase_shift = phi_spherical_tilted
    phase_shift = np.mod(phi_spherical_tilted, 2 * np.pi)
    phase_shift = phase_shift * 180 / np.pi
    return phase_shift

def horn_phase_weight(x_loc, y_loc):
    return phase_deg_to_weight(horn_phase_shift(x_loc, y_loc))

# # Plot the tilted spherical phase distribution in X-Y plane at Z = Z_const
# plt.figure(figsize=(8, 6))
# plt.contourf(X, Y, Phi_spherical_tilted, levels=50, cmap='jet')
# plt.colorbar(label="Phase (radians)")
# plt.xlabel("X-axis (m)")
# plt.ylabel("Y-axis (m)")
# plt.title(f"Tilted Spherical Wavefront Phase at Z={Z_const} m, Tilt={np.degrees(theta_t)}°")
# plt.show()


if __name__ == '__main__':
    plt.figure()
    plt.imshow((HORN_PHASE_SHIFT.T), extent=[X.min(), X.max(), Y.min(), Y.max()], clim=(-180, 180))
    plt.colorbar(label='Phase Shift (deg)')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.show()