import spi
import random_voltages
import time
import pna
import numpy as np
import json
import subprocess
import os
import signal


def write_voltages(v):
    file = "/home/ldantes/Desktop/data.csv"
    with open(file, "w") as f:
        for i in range(random_voltages.ROWS):
            for j in range(random_voltages.COLS):
                f.write(str(round(v[i*random_voltages.COLS+j], 2)))
                if (j != random_voltages.COLS-1):
                    f.write(",")
            f.write("\n")

if __name__ == '__main__':
    # v = random_voltages.get_random_voltages_1d()
    # write_voltages(v)
    # proc = subprocess.Popen(["python", "/home/ldantes/sathvik/spi.py"])
    # time.sleep(20)
    # proc.kill()
    # exit(0)


    voltages = []
    phases = []
    count = 0
    for v in range(0, 101, 5):
        curv = random_voltages.get_random_voltages_1d()
        for i in range(96):
            curv[i] = v
        voltages.append(v)
        write_voltages(curv)

        proc = subprocess.Popen(["python", "/home/ldantes/sathvik/spi.py"])

        time.sleep(30)

        # record here
        af = pna.get_af().tolist()
        p = np.round(np.degrees(np.angle(af)), 3).tolist()
        magnitudes = np.round(np.abs(af), 5).tolist()

        proc.kill()

        phases.append(p[150])

    print(voltages)
    print(phases)