"""
Plots expected information gain for a range of phis.

The ground truth is that the value of phi is 0.0 hence
we expect the value of expected information gain to be
maximum at this point.
"""
import numpy as np
import matplotlib.pyplot as plt

from numpy.testing import assert_equal
from toy_model_new import optimize_phi

def test_optimize_phi():
    # Setting phi=0.01 captures the least uncertainty about theta.
    phis = np.array([0.01, 0.1, 0.3 ,0.5, 1.0])
    best_phi, eigs = optimize_phi(phis, return_inf=True)
    worst_phi = phis[np.argmin(eigs)]
    assert_equal(best_phi, 0.01)
    assert_equal(worst_phi, 1.0)

#test_optimize_phi()
bins = np.linspace(-3.0, 6.0, 100)
bin_width = (bins[1] - bins[0]) / 2.0
X_s = (bins + bin_width)[:-1]

plt.clf()
phis = np.array([0.1, 0.6])
eigs, likelihoods, thetas = optimize_phi(phis, return_likelihoods=True)
for phi_ind, phi in enumerate(phis):
    for theta in [0.8, 1.0, 1.2]:
        closest_ind = np.argmin(np.abs(thetas[phi_ind] - theta))
        plt.plot(X_s, likelihoods[phi_ind][closest_ind])
        # plt.title("Likelihood ")
        plt.title("P(X | theta=%0.2f, phi=%0.2f)" % (theta, phi))
        plt.xlabel("X")
        plt.ylabel("Probablity")
        plt.show()


# phis = np.logspace(-2, 0, 10)
# best_phi, eigs = optimize_phi(phis, return_inf=True)
# objects = ["%.2f" % x for x in phis]
# y_pos = np.arange(len(objects))
# eigs = [ -x for x in eigs]
#
# plt.figure(figsize=(10,5))
# plt.bar(y_pos, eigs, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.xlabel('Phi')
# plt.ylabel('Expected Information Gain')
# plt.show()
