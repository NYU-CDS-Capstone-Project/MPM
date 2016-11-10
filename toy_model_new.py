import numpy as np

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
    """
    Returns P(X_i) given a tuple of n, bins as returned from
    np.histogram
    """
    n, bins = empirical_pdf
    # Generate samples to estimate the empirical distribution.
    bin_indices = np.searchsorted(bins, X) - 1

    # Clip values outside the interval.
    bin_indices[bin_indices == -1] = 0
    bin_indices[bin_indices == len(n)] = len(n) - 1
    P_X_given_theta = n[bin_indices]
    return P_X_given_theta


def expected_information_gain(thetas, X, phi, prior=None):
    """
    Calculates E(-H(P(theta | X, phi))) where
    each X_i is drawn iid from P(X | phi)

    P(theta | X_i, phi) is estimated by P(X_i | theta, phi) * prior

    Arguments
    ---------
    thetas - A set of plausible thetas.

    X - Data generated from P(X | phi)

    phi - Experimental setting

    prior - P(theta | phi)
    """
    if prior is None:
        prior = np.ones_like(thetas) / float(len(thetas))

    posteriors = np.zeros((len(thetas), len(X)))
    for theta_ind, theta in enumerate(thetas):
        emp_pdf = empirical_pdf(theta, phi) # here we do 1MM blackbox samples
        posteriors[theta_ind] = likelihood(X, emp_pdf) * prior[theta_ind]

    posteriors = posteriors / posteriors.sum(axis=0)
    log_posterior = safe_ln(posteriors)
    p_log_p = posteriors * log_posterior

    # Sum across thetas to get P(theta | X_i, phi)
    # Take average across X_i's to get expected information gain.
    entropies = np.sum(p_log_p, axis=0)
    return np.mean(entropies)


def optimize_phi(phis, return_inf=False):
    """
    Find the phi with maximum expected information gain.
    """
    phis = np.array(phis)
    rng = np.random.RandomState(0)

    # XXX: Use more intelligent step size.
    thetas = np.linspace(0.8, 1.2, 100)

    eigs = []
    for phi in phis:
        # Draw n_samples from P(X | phi)

        # Procedure:
        # 1. Draw n_thetas using a given prior:
        # 2. For each theta:
        #    Draw n_samples from P(X | theta_i, phi) from the black box.
        # 3. Average across theta_i's.

        # Assume uniform prior for now.
        # Draw 100 thetas
        n_samples = np.zeros((100, 1000))
        pick_thetas = thetas[rng.randint(0, len(thetas), len(thetas))]
        for theta_ind, theta in enumerate(pick_thetas):
            # call the black box for each theta, request 1000 samples,
            # store the samples for that theta in a row of the n_sample matrix
            n_samples[theta_ind] = black_box(1000, theta, phi, theta_ind)

        # Average samples across thetas
        # Equivalent to drawing from P(X | phi)
        data = n_samples.mean(axis=0)

        eig = expected_information_gain(thetas, data, phi)
        eigs.append(eig)
    best_phi = phis[np.argmax(eigs)]
    if return_inf:
        return best_phi, eigs
    return best_phi
