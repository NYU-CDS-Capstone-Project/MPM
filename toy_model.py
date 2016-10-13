import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

rng = np.random.RandomState(0)

def black_box(n_samples, theta=1.0, phi=0.2):
    """
    Black box for which we know the distribution follows normal
    with mean theta and std phi.
    """
    return phi * rng.randn(n_samples) + theta

def log_likelihood(X, theta, phi):
    """
    Gives likelihood of P(X )
    """
    samples = black_box(10**6, theta, phi)
    n, bins, _ = plt.hist(samples, 100)
    bin_indices = np.searchsorted(bins, X) - 1

    # Clip values outside the interval.
    bin_indices[bin_indices == -1] = 0
    bin_indices[bin_indices == len(n)] = len(n) - 1
    n_counts = n[bin_indices]

    P_X_theta = n_counts / np.sum(n_counts)
    return np.log(np.sum(P_X_theta))

def compute_posterior(thetas, phi, X):
    """
    Compute P(theta | phi, X)
    """
    posteriors = []
    for theta in thetas:

        # Find log(\prod_{i=1}^n P(X_i | t, phi)
        log_like = log_likelihood(X, theta, phi)

        # Log-Prior:
        # Assume P(theta | phi) is a gaussian with mean 1.0
        # and std phi
        prior = np.log(norm.pdf(theta, 1.0, phi))
        posteriors.append(log_like + prior)

    posteriors = np.array(posteriors)
    normalize = np.sum(posteriors)
    return np.sign(normalize) * posteriors / normalize

def optimize_phi(posteriors, thetas):
    best_entropy = np.sum(posteriors * np.exp(posteriors))
    phis = np.linspace(0.5, 1.5, 20)
    inf_gain = []

    for phi in phis:
        posteriors = compute_posterior(thetas, phi, samples)
        curr_entropy = np.sum(posteriors * np.exp(posteriors))
        inf_gain.append(best_entropy - curr_entropy)
    return phis[np.argmax(inf_gain)]

phi = 0.1
thetas = np.linspace(0.5, 1.5, 100)

for i in range(10):

    samples = black_box(1000, 1.0, phi)

    # Compute posterior.
    posteriors = compute_posterior(thetas, phi, samples)
    phi = optimize_phi(posteriors, thetas)
    print(phi)
