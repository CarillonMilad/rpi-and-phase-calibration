import numpy as np
from vars import *

# CUR_AF_AT_DIRECTED_POINT = 0
CUR_AF_AT_DIRECTED_POINTS = np.zeros(DIRECTED_POINTS, dtype=complex)

COST_MEASUREMENTS = 0
ITERATIONS = 10

def set_cost_measurements(val):
    global COST_MEASUREMENTS
    COST_MEASUREMENTS = val


MIN_DEG = -150
MAX_DEG = 150