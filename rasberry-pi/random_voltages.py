import random
import numpy as np

ROWS = 8
COLS = 12


def print_voltages(v):
    for i in range(ROWS):
        for j in range(COLS):
            print(round(v[i][j], 2), end='')
            if (j != COLS-1):
                print(',', end='')
        print()

def get_random_voltages():
    v = np.zeros((ROWS, COLS))
    for i in range(ROWS):
        for j in range(COLS):
            v[i, j] = random.uniform(0, 100)
    return v

def get_random_voltages_1d():
    return np.round(get_random_voltages().reshape((ROWS*COLS)), 3).tolist()

if __name__ == '__main__':
    v = get_random_voltages().tolist()
    v = np.zeros((ROWS, COLS))
    for i in range(ROWS):
        for j in range(COLS):
            v[i][j] = 20
    print_voltages(v)