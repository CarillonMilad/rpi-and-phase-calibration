import random

import horn_shifts
import model
from vars import *

def voltGivenPhaseInterp(phase):
    interpVolt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    interpPhase = [149, 146, 135, 110, 82, 51, 19, -19, -55, -85, -105, -126, -137, -142, -146, -149]
    return (np.interp(phase, interpPhase[::-1], interpVolt[::-1]))

def phaseGivenVoltInterp(volt):
    interpVolt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    interpPhase = [149, 146, 135, 110, 82, 51, 19, -19, -55, -85, -105, -126, -137, -142, -146, -149]
    return (np.interp(volt, interpVolt, interpPhase))


# ELEMENT_PATTERN_AT_DIRECTED_POINT = antenna_element_pattern(theta0/180*pi, phi0/180*pi,
#                                               cos_factor_theta,
#                                               cos_factor_phi,
#                                               max_gain_dBi=ep_max_gain_dBi)

# IN RADIANS
def af_at_point(phase_weights, theta1, phi1):
    # element_pattern = antenna_element_pattern(theta1, phi1,
    #                                           cos_factor_theta,
    #                                           cos_factor_phi,
    #                                           max_gain_dBi=ep_max_gain_dBi)
    # af_at_point = CUR_AF_AT_DIRECTED_POINT
    af_at_point = model.AF(theta1, phi1, x=X_vec, y=Y_vec, w=phase_weights, k=k)
    #print("AT", af_at_point)
    #print("DIFF", CUR_AF-af_at_point)
    #exit(0)
    # af_at_point *= element_pattern
    return abs(af_at_point)

def af_cmplx_at_directed_point(theta, phi, phase_weights):
    pw = phase_weights.copy()
    if (INCLUDE_HORN_PHASE_SHIFT):
        pw *= horn_shifts.HORN_PHASE_WEIGHT
    if (INCLUDE_RANDOM_PHASE):
        pw *= model.random_phase
    return model.AF(theta, phi, x=X_vec, y=Y_vec, w=pw, k=k)

def af_cmplx_at_directed_point_from_voltages(theta, phi, voltages):
    pw = phase_shift_deg_to_weight(phaseGivenVoltInterp(voltages))
    return af_cmplx_at_directed_point(theta, phi, pw)

def cost(phase_weights):
    pw = phase_weights.copy()
    if (INCLUDE_HORN_PHASE_SHIFT):
        pw *= horn_shifts.HORN_PHASE_WEIGHT
    if (INCLUDE_RANDOM_PHASE):
        pw *= model.random_phase
    # COST = -af_at_point(pw, theta0 * pi / 180, phi0 * pi / 180)
    # return COST

    af_at_pts = np.zeros(DIRECTED_POINTS)
    COST = 0
    for i in range(DIRECTED_POINTS):
        t = directed_theta[i]
        p = directed_phi[i]
        af_at_pts[i] = abs(af_at_point(pw, t * pi / 180, p * pi / 180))
        COST += -af_at_pts[i]
    af_avg = np.sum(af_at_pts) / DIRECTED_POINTS
    for i in range(DIRECTED_POINTS):
        COST += abs(af_avg - af_at_pts[i])
    return COST



def phase_shift_deg_to_weight(phase_shift_deg):
    return np.exp(1j * phase_shift_deg / 180.0 * pi)