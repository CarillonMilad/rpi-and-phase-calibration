import cost
import optimization
import optimization_vars
import numpy as np
import matplotlib.pyplot as plt
from vars import *

import random

import horn_shifts
import model
from circle_fit import extract_phase_from_AF_samples
from horn_shifts import phase_deg_to_weight
from model import wrap_angle
from vars import *
import time
import cost
import optimization_vars


def solve_phase_weights():
    phase_weights = np.zeros((Ny, Nx), dtype=complex)


    # fix [0, 0] as 0 phase shift
    phase_weights[0, 0] = phase_deg_to_weight(0)

    cost.set_af(phase_weights)

    for i in range(Ny):
        for j in range(Nx):
            if (i == 0 and j == 0):
                continue

            best = [1e9, 0] # cost, deg
            for deg in np.linspace(optimization_vars.MIN_DEG, optimization_vars.MAX_DEG, optimization_vars.ITERATIONS):

                old_phase = phase_weights[i, j]
                phase_weights[i, j] = phase_deg_to_weight(deg)
                new_phase = phase_weights[i, j]

                cost.upd_af(old_phase, new_phase, j, i)
                COST = cost.current_cost()
                # print(COST-cost.cost(phase_weights))
                best = min(best, [COST, deg])

                # print(round(COST, 2), end=' ')
            # print()

            old_phase = phase_weights[i, j]
            phase_weights[i, j] = phase_deg_to_weight(best[1])
            best_phase = phase_weights[i, j]
            cost.upd_af(old_phase, best_phase, j, i)

    return phase_weights



def voltGivenPhaseInterp(phase):
    interpVolt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    interpPhase = [149, 146, 135, 110, 82, 51, 19, -19, -55, -85, -105, -126, -137, -142, -146, -149]
    return (np.interp(phase, interpPhase[::-1], interpVolt[::-1]))

def phaseGivenVoltInterp(volt):
    interpVolt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    interpPhase = [149, 146, 135, 110, 82, 51, 19, -19, -55, -85, -105, -126, -137, -142, -146, -149]
    return (np.interp(volt, interpVolt, interpPhase))



def get_voltage_to_phase():
    UNIQUE_VOLTAGES = 21
    MIN_VOLTAGE = 0
    MAX_VOLTAGE = 100

    voltages = np.zeros((Ny, Nx))
    voltages.fill(MAX_VOLTAGE)


    voltage_and_phase = np.zeros((Ny, Nx, 2, UNIQUE_VOLTAGES))  # voltage then phase

    theta = 0
    phi = 0

    for i in range(Ny):
        for j in range(Nx):
            # Prepare to collect AF samples for this element
            af_samples = np.zeros(UNIQUE_VOLTAGES, dtype=complex)
            voltage_sweep = np.linspace(MIN_VOLTAGE, MAX_VOLTAGE, UNIQUE_VOLTAGES)

            for idx, v in enumerate(voltage_sweep):
                voltages[i, j] = v
                phase_weights = phase_deg_to_weight(phaseGivenVoltInterp(voltages))
                af = cost.af_cmplx_at_directed_point(theta0, phi0, phase_weights)
                af_samples[idx] = af

            # Extract unwrapped phase vs voltage using circle fit
            phases, _ = extract_phase_from_AF_samples(af_samples)

            # Store voltages and extracted phases
            voltage_and_phase[i, j, 0, :] = voltage_sweep
            voltage_and_phase[i, j, 1, :] = phases

            voltages[i, j] = random.uniform(MIN_VOLTAGE, MAX_VOLTAGE)
    return voltage_and_phase


def solve_voltages_and_phases():
    UNIQUE_VOLTAGES = 21
    MIN_VOLTAGE = 0
    MAX_VOLTAGE = 100

    voltage_sweep = np.linspace(MIN_VOLTAGE, MAX_VOLTAGE, UNIQUE_VOLTAGES)

    voltages = np.full((Ny, Nx), MAX_VOLTAGE)

    voltage_and_phase = np.zeros((Ny, Nx, 2, UNIQUE_VOLTAGES))
    best_phases = np.zeros((Ny, Nx))

    for i in range(Ny):
        for j in range(Nx):
            af_samples = np.zeros(UNIQUE_VOLTAGES, dtype=complex)

            # use current voltages (may include updated values from earlier elements)
            for idx, v in enumerate(voltage_sweep):
                voltages[i, j] = v
                phase_weights = phase_deg_to_weight(phaseGivenVoltInterp(voltages))
                af_samples[idx] = cost.af_cmplx_at_directed_point(theta0, phi0, phase_weights)


            phases, (cx, cy, r) = extract_phase_from_AF_samples(af_samples)
            voltage_and_phase[i, j, 0, :] = voltage_sweep
            voltage_and_phase[i, j, 1, :] = phases


            center = cx + 1j*cy
            best_phase = np.angle(center)


            b = [0, 0]
            for idx, p in enumerate(phases):
                b = max(b, [abs(af_samples[idx]), p])

            best_phases[i, j] = b[1]
            print(best_phase*180/pi, b[1]*180/pi)
            voltages[i, j] = np.interp(best_phase, phases, voltage_sweep)

    return voltage_and_phase, best_phases, voltages


def test_voltage_to_phase():
    voltage_and_phase = get_voltage_to_phase()
    for i in range(Ny):
        for j in range(Nx):
            v = voltage_and_phase[i, j, 0]
            p = np.degrees(np.unwrap(voltage_and_phase[i, j, 1]))
            print(v)
            print(p)
            plt.plot(v, p, linestyle='-', marker='o', color='r')
            plt.xlabel('Voltage (V)')
            plt.ylabel('Phase (deg)')
            plt.show()

    interpVolt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    interpPhase = [149, 146, 135, 110, 82, 51, 19, -19, -55, -85, -105, -126, -137, -142, -146, -149]
    plt.plot(interpVolt, interpPhase, linestyle='-', marker='o', color='r')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Phase (deg)')
    plt.show()

def plot_cost_vs_iterations():

    iter_vals = []
    costs = []
    for iter in range(1, 31, 1):
        optimization_vars.ITERATIONS = iter
        pw = get_phase_weights()
        iter_vals.append(iter)
        costs.append(cost.cost(pw))

    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.grid(True)
    plt.ylim(ymin=-200, ymax=0)
    plt.plot(iter_vals, costs, 'o')
    plt.show()




if __name__ == '__main__':
    test_voltage_to_phase()
    # plot_cost_vs_iterations()
