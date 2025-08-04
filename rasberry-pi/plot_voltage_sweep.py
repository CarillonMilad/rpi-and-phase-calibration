
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    voltages = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

    phases = [-117.415, -120.336, -128.246, -138.779, -147.249, -153.868, -161.151, -164.505, -167.909, -172.642, -175.64, -176.907, -178.132, 178.975, 178.799, 175.809, 177.116, 178.499, 178.228, 177.975, 176.937]


    plt.plot(voltages, phases, marker='o', linestyle='-')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Phase (deg)')
    plt.show()
