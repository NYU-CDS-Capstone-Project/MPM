import numpy as np

def entropy(p_thetas):
    entropies = np.zeros_like(p_thetas)
    mask = p_thetas > 0
    entropies[mask] = p_thetas[mask] * np.log(p_thetas[mask])
    return -np.sum(entropies)

def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

def black_box(n_samples, theta=1.0, phi=0.2, random_state=None):
    """
    Black box for which we know the distribution follows normal
    with mean theta and std phi.
    """
    rng = np.random.RandomState(random_state)
    return phi * rng.randn(n_samples) + theta

def empirical_pdf(theta, phi):
    samples = black_box(10**6, theta, phi)
    return np.histogram(samples, 1000, density=True)

def likelihood(X, empirical_pdf):
    n, bins = empirical_pdf
    # Generate samples to estimate the empirical distribution.
    bin_indices = np.searchsorted(bins, X) - 1

    # Clip values outside the interval.
    bin_indices[bin_indices == -1] = 0
    bin_indices[bin_indices == len(n)] = len(n) - 1
    P_X_given_theta = n[bin_indices]
    return P_X_given_theta

def expected_information_gain(thetas, X, phi, prior=None):
    if prior is None:
        prior = np.ones_like(thetas) / float(len(thetas))

    posteriors = np.zeros_like(thetas)
    log_likelihoods = np.zeros_like(thetas)

    posteriors = np.zeros((len(thetas), len(X)))
    for theta_ind, theta in enumerate(thetas):
        emp_pdf = empirical_pdf(theta, phi)
        posteriors[theta_ind] = likelihood(X, emp_pdf) * prior[theta_ind]

    posteriors = posteriors / posteriors.sum(axis=0)
    log_posterior = safe_ln(posteriors)
    p_log_p = posteriors * log_posterior

    entropies = np.sum(p_log_p, axis=0)
    return np.mean(entropies)
