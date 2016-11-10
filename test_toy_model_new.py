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

phis = np.logspace(-2, 0, 10)
best_phi, eigs = optimize_phi(phis, return_inf=True)
objects = ["%.2f" % x for x in phis]
y_pos = np.arange(len(objects))
eigs = [ -x for x in eigs]

plt.figure(figsize=(10,5))
plt.bar(y_pos, eigs, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.xlabel('Phi')
plt.ylabel('Expected Information Gain')
plt.show()
