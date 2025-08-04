import json
import numpy as np

with open("data1.json", "r") as f:
    data = json.load(f)

V = data["voltages"]
P = data["phases"]
M = data["magnitudes"]


for j in range(len(M)):
    mx = [0, 0]
    for i in range(len(M[j])):
        mx = max(mx, [M[j][i],i])

    print(mx)


for i in range(len(V[0])):
    print(V[0][i] - V[1][i])