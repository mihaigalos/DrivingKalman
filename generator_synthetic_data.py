from random import randrange, uniform
import numpy as np
from numpy import sin, sqrt, cumsum
import matplotlib.pyplot as plt


class SyntheticData:
    def __init__(self):

        dt = 0.01
        max = 70
        self.t = np.linspace(0, max, max / dt)

        accX_var_best = 0.0005
        accX_var_good = 0.0007
        accX_var_worst = 0.001

        accX_ref_noise = np.random.rand(len(self.t)) * sqrt(accX_var_best)
        accX_good_noise = np.random.rand(len(self.t)) * sqrt(accX_var_good)
        accX_worst_noise = np.random.rand(len(self.t)) * sqrt(accX_var_worst)

        accX_basesignal = sin(0.3 * self.t) + 0.5 * sin(0.04 * self.t)

        self.accX_ref = accX_basesignal + accX_ref_noise
        self.velX_ref = cumsum(self.accX_ref) * dt
        self.distX_ref = cumsum(self.velX_ref) * dt

        accX_good_offset = 0.001 + 0.0004 * sin(0.05 * self.t)

        self.accX_good = accX_basesignal + accX_good_noise + accX_good_offset
        self.velX_good = cumsum(self.accX_good) * dt
        self.distX_good = cumsum(self.velX_good) * dt

        accX_worst_offset = -0.08 + 0.004 * sin(0.07 * self.t)

        self.accX_worst = accX_basesignal + accX_worst_noise + accX_worst_offset
        self.velX_worst = cumsum(self.accX_worst) * dt
        self.distX_worst = cumsum(self.velX_worst) * dt

    def plot(self):
        f, axarr = plt.subplots(3, sharex=True)

        f.suptitle("Synthetic data for KalmanFilter evaluation")

        axarr[0].plot(self.t, self.accX_ref, "g-", label="Ground Truth")
        axarr[0].plot(self.t, self.accX_good, "b-", label="Good")
        axarr[0].plot(self.t, self.accX_worst, "r-", label="Worst")
        axarr[0].set_title('Acceleration X')
        axarr[0].grid()
        axarr[0].legend()

        axarr[1].plot(self.t, self.velX_ref, "g-", label="Ground Truth")
        axarr[1].plot(self.t, self.velX_good, "b-", label="Good")
        axarr[1].plot(self.t, self.velX_worst, "r-", label="Worst")
        axarr[1].set_title('Velocity X')
        axarr[1].grid()
        axarr[1].legend()

        axarr[2].plot(self.t, self.distX_ref, "g-", label="Ground Truth")
        axarr[2].plot(self.t, self.distX_good, "b-", label="Good")
        axarr[2].plot(self.t, self.distX_worst, "r-", label="Worst")
        axarr[2].set_title('Position X')
        axarr[2].grid()
        axarr[2].legend()

        plt.show()


SyntheticData().plot()
