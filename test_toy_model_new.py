"""
Plots expected information gain for a range of phis.

The ground truth is that the value of phi is 0.0 hence
we expect the value of expected information gain to be
maximum at this point.
"""
import numpy as np
import matplotlib.pyplot as plt
from toy_model_new import black_box
from toy_model_new import expected_information_gain
from toy_model_new import optimize_phi

thetas = np.linspace(0.8, 1.2, 100)
X = black_box(theta=1.0, phi=0.0, n_samples=1000)
# phis = np.linspace(-0.5, 0.5, 100)
eigs = optimize_phi([0.01, 0.1, 0.3 ,0.5])
# eigs = []
#
# for phi in phis:
#     print(phi)
#     eig = expected_information_gain(thetas, X, phi)
#     eigs.append(eig)
#
# plt.plot(phis, eigs)
# plt.xlabel("Phis")
# plt.ylabel("Expected information gain.")
# plt.show()
