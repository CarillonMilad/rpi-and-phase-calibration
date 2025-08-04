import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
from vars import *
import horn_shifts
import model
import cost

# steering
THETA_DEG = theta0
PHI_DEG = phi0
THETA_RAD = theta0/180*pi
PHI_RAD = phi0/180*pi
STEER_SHIFTS = np.exp(-1j * k * (X_vec * np.sin(THETA_RAD) * np.cos(PHI_RAD) + Y_vec * np.sin(THETA_RAD) * np.sin(PHI_RAD)))
STEER_SHIFTS = torch.tensor(STEER_SHIFTS)

EFFICIENCY = 0.6

# array
ROWS, COLS = Ny, Nx
N = ROWS * COLS
V_MIN, V_MAX = 0.0, 1.0
M = 1000 # random voltage samples


# sigmoid
PHASE_RANGE = 300 * math.pi / 180
SLOPE_INIT = -9.5610
CENTER_INIT = 0.3068
VERT_SHIFT_INIT = -2.5543



# learning
LEARNING_RATE = 0.01
EPOCHS = 10000
PUNISH_POWER = 2



class PhaseModel(nn.Module):
    def __init__(self, num_elements):
        super().__init__()
        self.vert_shift = nn.Parameter(torch.tensor([VERT_SHIFT_INIT for _ in range(num_elements)]))  # [N]
        self.slope = nn.Parameter(torch.tensor([SLOPE_INIT for _ in range(num_elements)]))     # [N]
        self.center = nn.Parameter(torch.tensor([CENTER_INIT for _ in range(num_elements)]))   # [N]

    def forward(self, V):
        """
        Args:
            V: [M, N] voltage inputs
        Returns:
            phi: [M, N] predicted phases
        """
        V = V.unsqueeze(-1)                              # [M, N, 1]
        vert_shift = self.vert_shift.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        slope = self.slope.unsqueeze(0).unsqueeze(-1)            # [1, N, 1]
        center = self.center.unsqueeze(0).unsqueeze(-1)          # [1, N, 1]

        phi = vert_shift + PHASE_RANGE / (1 + torch.exp(-slope * (V - center)))  # [M, N, 1]
        return phi.squeeze(-1)                       # [M, N]

def train_phase_model(voltage_data, measured_phases):
    model = PhaseModel(N)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100)
    # reduce learning rate by xfactor (x0.5) with patience=1000 epochs of no improvement
    loss_history = []

    true_phasor = torch.exp(1j * measured_phases)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        phi_per_element_pred = model(voltage_data)                          # [M, N]
        af_pred = torch.sum(torch.exp(1j * phi_per_element_pred) * STEER_SHIFTS, dim=1)  # [M]

        pred_phasor = af_pred / torch.abs(af_pred)  # normalize to unit magnitude
        loss = torch.mean(torch.abs(pred_phasor - true_phasor) ** PUNISH_POWER)

        loss.backward()
        optimizer.step()
        scheduler.step(loss)  # Reduce LR on plateau

        loss_history.append(loss.item())
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, LR = {current_lr:.6e}")
    return model, loss_history


def voltage_to_phase(params, v):
    vert_shift, slope, center = params
    return vert_shift + PHASE_RANGE / (1 + np.exp(-slope * (v - center)))

def phase_to_voltage(params, p):
    vert_shift, slope, center = params

    PHASE_MIN = voltage_to_phase(params, 1)
    PHASE_MAX = voltage_to_phase(params, 0)
    assert(PHASE_MIN < PHASE_MAX)
    while p < PHASE_MIN:
        p += 2 * math.pi
    while p > PHASE_MAX:
        p -= 2 * math.pi

    if (PHASE_RANGE / (p - vert_shift) - 1 <= 0):
        return 0

    v = center - (1 / slope) * np.log(PHASE_RANGE / (p - vert_shift) - 1)
    v = max(v, 0)
    v = min(v, 1)
    return v * 100  # scale to DAC range


def noise():
    return random.gauss(0, 1)



if __name__ == '__main__':

    voltages = np.random.uniform(V_MIN, V_MAX, size=(M, N))
    measured_af = np.zeros(M, dtype=complex)
    measured_phase = np.zeros(M, dtype=float)

    for i in range(M):
        v = voltages[i].reshape((ROWS, COLS))
        af = cost.af_cmplx_at_directed_point_from_voltages(THETA_RAD, PHI_RAD, v * 100)
        measured_phase[i] = np.angle(af) + (noise()/180*pi)
        magnitude = 10**np.round(np.log10(abs(af)), 2) + (noise()/5)
        af = magnitude * (af/abs(af))
        af = np.round(af, 2) + (noise()+1j*noise())
        af *= EFFICIENCY
        measured_af[i] = af
        print(measured_phase[i])

    voltage_data = torch.tensor(voltages, dtype=torch.float32)
    measured_fields = torch.tensor(measured_af, dtype=torch.cfloat)
    measured_phases_tensor = torch.tensor(measured_phase, dtype=torch.float32)

    # Train model
    phase_model, losses = train_phase_model(voltage_data, measured_phases_tensor)
    vert_shifts = phase_model.vert_shift.detach().numpy()
    slopes = phase_model.slope.detach().numpy()
    centers = phase_model.center.detach().numpy()

    print("Sample parameters:")
    print("vert_shift:", vert_shifts[50])
    print("slope:", slopes[50])
    print("center:", centers[50])

    # Reconstruct voltage map from ideal phases
    phase_weights_correct = model.steering_vector(k=k, xv=X, yv=Y, theta_deg=THETA_DEG, phi_deg=PHI_DEG)
    phase_deg_correct = np.degrees(np.angle(phase_weights_correct))
    print(phase_deg_correct)
    model.plot_phase_shifts(phase_deg_correct)
    model.plot_phase_shifts(np.degrees(np.angle(phase_weights_correct*horn_shifts.HORN_PHASE_WEIGHT)))

    my_voltages = np.zeros((ROWS, COLS))
    for i in range(ROWS):
        for j in range(COLS):
            idx = i * COLS + j
            phase = np.angle(phase_weights_correct[i][j])
            params = (vert_shifts[idx], slopes[idx], centers[idx])
            my_voltages[i][j] = phase_to_voltage(params, phase)

    print("Voltages:")
    print(my_voltages)

    my_phase_deg = cost.phaseGivenVoltInterp(my_voltages)
    my_phase_weights = cost.phase_shift_deg_to_weight(my_phase_deg)
    model.plot_phase_shifts(my_phase_deg)
    print("COST: ", cost.cost(my_phase_weights))
    model.plot_pattern(my_phase_weights)