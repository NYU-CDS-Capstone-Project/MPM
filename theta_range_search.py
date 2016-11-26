import numpy as np

import matplotlib.pyplot as plt
from sys import exit
def safe_ln(x, minval=0.00001):
    return np.log(x.clip(min=minval))

def black_box(n_samples, theta=1.0, phi=0.2, random_state=None):
    """
    Black box for which we know the distribution follows normal
    with mean theta and std phi.
    """
    phi = 2 + np.cos(phi)
    rng = np.random.RandomState(random_state)
    return phi * rng.randn(n_samples) + theta


def log_likelihood(X, theta, phi):
    """
    Gives likelihood of P(X | theta, phi)

    Parameters
    ----------
    X - shape(n_samples,)
        Samples drawn from the black box.

    theta - float
        Parameter value.

    phi - float
        Experimental setting.

    Returns
    -------
    likelihood - float
        \prod_{i=1}^n P(X_i | theta, phi)
    """
    # Generate samples to estimate the empirical distribution.
    samples = black_box(10**6, theta, phi)
    n, bins = np.histogram(samples, 1000, density=True)
    bin_indices = np.searchsorted(bins, X) - 1

    # Clip values outside the interval.
    bin_indices[bin_indices == -1] = 0
    bin_indices[bin_indices == len(n)] = len(n) - 1
    n_counts = n[bin_indices]
    P_X_given_theta = n_counts / np.sum(n_counts)
    return np.sum(safe_ln(P_X_given_theta))


def compute_log_posterior(thetas, phi, X, log_prior, run_iter="init", phi_iter="init", exp_iter="init"):
    """
    Compute P(theta | phi, X) = P(theta | phi) * P(X | theta, phi)

    Parameters
    ----------
    thetas - shape=(n_thetas,)
        List of permissible values of thetas.

    phi - float
        Experimental setting.

    X - shape=(n_samples,)
        Samples drawn from the black box.

    log_prior - float
        log(P(theta))

    Returns
    -------
    log_posterior - shape=(n_thetas,)
        Log posterior.
    """
    log_posterior = np.empty_like(thetas)
    log_likelihoods = np.empty_like(thetas)

    for i, theta in enumerate(thetas):
        # Find log(\prod_{i=1}^n P(X_i | t, phi)
        log_like = log_likelihood(X, theta, phi)
        log_likelihoods[i] = log_like
        #print log_like 

    log_posterior = log_likelihoods + log_prior

    max_log_likelihood = max(log_likelihoods)
    log_likelihoods -= max_log_likelihood
    likelihood = np.exp(log_likelihoods)
    sum_likelihood = np.sum(likelihood)
    likelihood = likelihood / np.sum(likelihood)

    max_log_prior = max(log_prior)
    log_prior -= max_log_prior
    prior = np.exp(log_prior)
    prior = prior / np.sum(prior)

    product = prior * likelihood
    posterior =  product / np.sum(product)
    log_posterior = safe_ln(posterior)

    return log_posterior

phi_real = 0.1
theta_true = 1.0
real_data = black_box(100, theta_true, phi_real, 1)

semiInterval=18
numberOfSteps = 100

minTheta = -semiInterval
maxTheta = semiInterval
thetas = np.linspace(minTheta, maxTheta, numberOfSteps)
log_prior = safe_ln(np.ones_like(thetas) / thetas.shape[0])
log_posterior = compute_log_posterior(thetas, phi_real, real_data, log_prior)
posterior = np.exp(log_posterior)
print posterior
plt.plot(thetas,posterior)
plt.show()


