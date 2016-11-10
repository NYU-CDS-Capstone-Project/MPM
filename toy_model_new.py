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

def empirical_pdf(theta, phi, bins=1000):
    samples = black_box(10**6, theta, phi)
    return np.histogram(samples, bins, density=True)

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

    X - P(X | phi)

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
    n_thetas = 100
    thetas = np.linspace(0.8, 1.2, n_thetas)

    eigs = []
    for phi in phis:
        # Draw n_samples from P(X | phi)

        # Procedure:
        # 1. Draw n_thetas using a given prior:
        # 2. For each theta:
        #    Draw n_samples from P(X | theta_i, phi) from the black box.
        # 3. Average across theta_i's.

        # Assume uniform prior for now.
        # Draw 100 thetas uniformly from the given range.
        pick_thetas = thetas[rng.randint(0, len(thetas), len(thetas))]

        # Set bins to a reasonable range to compute the empirical pdf
        # of X given theta and phi.
        bins = np.linspace(-5.0, 5.0, 100)
        bin_width = (bins[1] - bins[0]) / 2.0
        X_s = (bins + bin_width)[:-1]

        P_X_given_theta_phi = np.zeros((n_thetas, len(bins) - 1))
        for theta_ind, theta in enumerate(pick_thetas):
            # compute empirical P(X | theta, phi) for these bins
            P_X_given_theta_phi[theta_ind] = empirical_pdf(theta, phi, bins)[0]

        # Compute P(X | phi) by averaging out P(X, theta | phi) across the
        # sampled values of theta.
        P_X_given_phi = np.mean(P_X_given_theta_phi, axis=0)
        P_X_given_phi /= np.sum(P_X_given_phi)

        # Since the bins are now discretized, sample 10000 values from
        # a multinomial with probabilities P_X_given_phi.
        data = np.repeat(X_s, rng.multinomial(10000, P_X_given_phi, size=1)[0])
        eig = expected_information_gain(thetas, data, phi)
        eigs.append(eig)
    best_phi = phis[np.argmax(eigs)]
    if return_inf:
        return best_phi, eigs
    return best_phi
