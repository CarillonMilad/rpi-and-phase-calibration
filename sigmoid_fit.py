import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

if __name__ == '__main__':
    # Your data
    interpVolt = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100], dtype=float)
    interpPhase_deg = np.array([149, 146, 135, 110, 82, 51, 19, -19, -55, -85, -105, -126, -137, -142, -146, -149], dtype=float)

    # Convert degrees to radians
    interpVolt /= 100.0
    interpPhase_rad = np.deg2rad(interpPhase_deg)

    # Define sigmoid function
    def sigmoid(x, A, B, C, D):
        return A + B / (1 + np.exp(-C * (x - D)))

    # Initial guess for parameters
    p0 = [0, 0, 0, 0]

    # Curve fitting
    params, covariance = curve_fit(sigmoid, interpVolt, interpPhase_rad, p0=p0)

    # Extract fitted params
    A, B, C, D = params
    print(f"Fitted params:\nA={A:.4f}, B={B:.4f}, C={C:.4f}, D={D:.4f}")

    # Plot original data and fit
    v_fit = np.linspace(0, 1, 300)
    phi_fit = sigmoid(v_fit, *params)

    plt.figure(figsize=(8, 5))
    plt.plot(interpVolt, interpPhase_rad, 'o', label="Original Data (rad)")
    plt.plot(v_fit, phi_fit, '-', label="Sigmoid Fit")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Phase (radians)")
    plt.title("Voltage to Phase: Sigmoid Fit")
    plt.grid(True)
    plt.legend()
    plt.show()
