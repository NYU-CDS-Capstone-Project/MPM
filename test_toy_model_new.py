"""
Plots expected information gain for a range of phis.

The ground truth is that the value of phi is 0.0 hence
we expect the value of expected information gain to be
maximum at this point.
"""
import numpy as np
import matplotlib.pyplot as plt

from numpy.testing import assert_equal
# from toy_model_new import black_box
# # from toy_model_new import expected_information_gain
from toy_model_new import optimize_phi

def test_optimize_phi():
    # Setting phi=0.01 captures the least uncertainty about theta.
    phis = np.array([0.01, 0.1, 0.3 ,0.5, 1.0])
    best_phi, eigs = optimize_phi(phis, return_inf=True)
    worst_phi = phis[np.argmin(eigs)]
    assert_equal(best_phi, 0.01)
    assert_equal(worst_phi, 1.0)
