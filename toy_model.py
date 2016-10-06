import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

rng = np.random.RandomState(0)

def black_box(n_samples, theta=1.0, phi=0.2):
    return phi * rng.randn(n_samples) + theta

def prior(theta, phi=0.2):
    """P(theta | phi) follows a gaussian with std phi"""
    return norm.pdf(theta, 0.0, phi)

def posterior(thetas, phi):
    temp = []
    for t in thetas:
        likelihood = np.log(norm.pdf(samples, t, phi))

        # Log-Prior:
        # Assume P(theta | phi) is a gaussian with mean 0.0
        # and std phi
        prior = np.log(norm.pdf(t, 0.0, phi))
        temp.append(np.sum(likelihood) + prior)

    return np.asarray(temp)

samples = black_box(n_samples=1000)
phis = np.linspace(0.1, 1.0, 1000)
entropies = []
for phi in phis:
    print(phi)
    thetas = np.linspace(0.5, 1.5, 1000)
    posteriors = posterior(thetas, phi)
    entropies.append(-np.sum(posteriors * np.exp(posteriors)))

print(phis[np.argmin(entropies)])
