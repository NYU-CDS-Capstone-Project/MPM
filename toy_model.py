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

    P_X_given_theta = n_counts / np.sum(n_counts)
    return np.sum(np.log(P_X_given_theta))

def compute_posterior(thetas, phi, X, priors):
    """
    Compute P(theta | phi, X)
    """
    posteriors = []
    
    for i in range(len(thetas)):
        # Find log(\prod_{i=1}^n P(X_i | t, phi)
        log_like = log_likelihood(X, thetas[i], phi)

        # Log-Prior:
        # Assume P(theta | phi) is a gaussian with mean 1.0
        # and std phi
        prior = np.log(priors[i])
        posteriors.append(log_like + prior)

    posteriors = np.array(posteriors)
    normalize = np.sum(posteriors)
    return np.sign(normalize) * posteriors / normalize

def optimize_phi(posteriors, thetas):
    # entropy of posteriors
    best_entropy = np.sum(posteriors * np.exp(posteriors))

    phis = np.linspace(0.5, 1.5, 20)
    inf_gain = []

    for phi in phis:
        posteriors = compute_posterior(thetas, phi, samples, posteriors)
        curr_entropy = np.sum(posteriors * np.exp(posteriors))
        inf_gain.append(best_entropy - curr_entropy)
    return phis[np.argmax(inf_gain)]


phi = 0.1
# generate a set of plausible thetas
thetas = np.linspace(0.5, 1.5, 100)

for i in range(10):
    # generate the initial data sample (x_1, x_2, ..., x_1000)
    samples = black_box(1000, 1.0, phi)

    # Compute posterior.
    priors = [norm.pdf(theta, 1.0, 1.0) for theta in thetas]
    posterior = compute_posterior(thetas, phi, samples, priors)
    phi = optimize_phi(posterior, thetas)
    print(phi)
