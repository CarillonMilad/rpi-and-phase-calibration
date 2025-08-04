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

    with open("data1.json", "w") as f:
        f.write("")

    all_voltages = []
    all_phases = []
    all_magnitudes = []

    count = 0
    while (1):
        v = random_voltages.get_random_voltages_1d()
        write_voltages(v)
        proc = subprocess.Popen(["python", "/home/ldantes/sathvik/spi.py"])

        time.sleep(30)

        # record here
        af = pna.get_af().tolist()
        phases = np.round(np.degrees(np.angle(af)), 3).tolist()
        magnitudes = np.round(np.abs(af), 5).tolist()

        proc.kill()

        print(phases[4])

        all_voltages.append(v)
        all_phases.append(phases)
        all_magnitudes.append(magnitudes)

        with open("data1.json", "r") as src, open("data2.json", "w") as dst:
            for line in src:
                dst.write(line)

        with open("data1.json", "w") as f:
            data = {
                "voltages": all_voltages,
                "phases": all_phases,
                "magnitudes": all_magnitudes
            }
            json.dump(data, f)

        print(count)
        count += 1

        if (count == 1000):
            break

    pna.instr.close()
    pna.rm.close()