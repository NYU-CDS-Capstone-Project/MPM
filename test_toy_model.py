import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

from toy_model import black_box
from toy_model import log_likelihood

def test_log_likelihood():
    """
    Test that maximum value of likeli
    """
    # Generate X from a distribution knowing the true value
    # of theta and phi
    true_theta = 1.0
    true_phi = 0.1
    toy_data = black_box(1000, theta=1.0, phi=0.1, random_state=0)

    likelihoods = []
    thetas = np.linspace(0.5, 1.5, 100)
    for theta in thetas:
        curr_likelihood = log_likelihood(toy_data, theta, true_phi)
        likelihoods.append(curr_likelihood)

    assert_allclose(
        [thetas[np.argmax(likelihoods)]], [true_theta], rtol=1e-2)
