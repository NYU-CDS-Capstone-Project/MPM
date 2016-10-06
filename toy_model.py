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

def posterior(thetas, phi, X):
    """
    Compute P(theta | phi, X)
    """
    posteriors = []
    for t in thetas:
        # Find log(\prod_{i=1}^n P(X_i | t, phi)
        likelihood = np.sum(np.log(norm.pdf(X, t, phi)))

        # Log-Prior:
        # Assume P(theta | phi) is a gaussian with mean 0.0
        # and std phi
        prior = np.log(norm.pdf(t, 0.0, phi))
        posteriors.append(likelihood + prior)

    return np.asarray(posteriors)

samples = black_box(n_samples=1000)
phis = np.linspace(0.1, 1.0, 1000)
thetas = np.linspace(0.5, 1.5, 1000)
entropies = []
for phi in phis:
    print(phi)
    posteriors = posterior(thetas, phi, samples)
    entropies.append(-np.sum(posteriors * np.exp(posteriors)))

# Find the setting of phi for which entropy over the distribution
# of theta is minimum.
print(phis[np.argmin(entropies)])
