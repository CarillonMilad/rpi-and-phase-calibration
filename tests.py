from vars import *
import model
from optimization import *

def af_at_point_correct(phase_weights, theta1, phi1):
    array_factor = np.zeros((len(theta), len(phi)), dtype=complex)

    for i, thi in enumerate(theta):
        for j, phj in enumerate(phi):
            array_factor[i, j] = 1 * model.AF(thi, phj, x=X_vec, y=Y_vec, w=phase_weights, k=k)
            # array_factor[i, j] = element_pattern[i, j] * AF(thi, phj, x=X_vec, y=Y_vec, w=phase_weights, k=k)
    return abs(array_factor[theta1+90, phi1+90])



def test_af_at_point():
    for p in range(-90, 91):
        for t in range(-90, 91):
            print(af_at_point_correct(phase_weights_correct, t, p) - cost.af_at_point(phase_weights_correct, t/180.0*pi, p/180.0*pi))


def plot_af_at_all_points():
    phase_weights_correct = model.steering_vector(k=k,
                                            xv=X,
                                            yv=Y,
                                            theta_deg=theta0,
                                            phi_deg=phi0)
    af = np.zeros((181, 181))
    for i in range(181):
        for j in range(181):
            af[i][j] = (model.af_at_point(phase_weights_correct, (i - 90) * pi / 180, (j - 90) * pi / 180))
    print(af)
    model.plot_stuff(af)


def print_phase_stuff(phase_stuff):
    for r in phase_stuff:
        for x in r:
            d = int(x)
            if (d >= 0):
                print("+", end='')
            print(d, end=' ')

            if (-10 < d and d < 10):
                print("  ", end='')
            elif (-100 < d and d < 100):
                print(" ", end='')
        print()