import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

rng = np.random.RandomState(0)

def safe_ln(x, minval=0.0000000001):
    return np.log(x.clip(min=minval))

def black_box(n_samples, theta=1.0, phi=0.2, random_state=None):
    """
    Black box for which we know the distribution follows normal
    with mean theta and std phi.
    """
    rng = np.random.RandomState(random_state)
    return phi * rng.randn(n_samples) + theta

def log_likelihood(X, theta, phi):
    """
    Gives likelihood of P(X )
    """
    samples = black_box(10**6, theta, phi)
    n, bins, _ = plt.hist(samples, 100, normed = True)
    bin_indices = np.searchsorted(bins, X) - 1

    # Clip values outside the interval.
    bin_indices[bin_indices == -1] = 0
    bin_indices[bin_indices == len(n)] = len(n) - 1
    n_counts = n[bin_indices]
    P_X_given_theta = n_counts / np.sum(n_counts)
    plt.clf()
    return np.sum(safe_ln(P_X_given_theta))

def compute_log_posterior(thetas, phi, X, prior):
    """
    Compute P(theta | phi, X)
    """
    log_posterior = []
    log_likelihoods = []
    log_prior = np.log(prior)

    for i, theta in enumerate(thetas):
        # Find log(\prod_{i=1}^n P(X_i | t, phi)
        log_like = log_likelihood(X, theta, phi)

        log_posterior.append(log_like + log_prior[i])
        log_likelihoods.append(log_like)

    # plt.plot(thetas, log_likelihoods)
    # plt.show()
    #
    # plt.clf()
    # plt.plot(thetas, log_posterior)
    # plt.show()

    # print(log_likelihoods)
    # print(thetas[np.argmax(log_posterior)])

    return log_posterior

def optimize_phi(posterior, thetas):
    # entropy of posterior
    best_entropy = np.sum(posterior * np.log(posterior))

    phis = np.linspace(0.5, 1.5, 20)
    mean_inf_gains = []
    best_phys = []
    N_toys = 10
    for phi in phis:
        inf_gain = []

        for i in range(N_toys):
            print("optimize_phi " + str(i))
            theta_best = np.argmax(posterior)
            toy_data = black_box(1000, theta_best, phi, i)
            posterior = compute_log_posterior(thetas, phi, toy_data, posterior)
            curr_entropy = np.sum(posterior * np.log(posterior))
            inf_gain.append(best_entropy - curr_entropy)
            best_phys.append(phis[np.argmax(inf_gain)])
        inf_gains.append(np.mean(inf_gain))
    return phis[np.argmax(inf_gains)]


def do_real_experiments():
    """
    Estimate the true value of phi for an experiment.
    """
    true_phi = 0.1
    true_theta = 1.0

    # generate a set of plausible thetas
    thetas = np.linspace(0.5, 1.5, 100)

    # Generate 1000 samples from the black box.
    real_data =  black_box(1000, true_theta, true_phi, 0)

    # Prior:
    # Assume P(theta | phi) is a gaussian with mean 1.0
    # and std 0.1
    prior = norm.pdf(thetas, 1.0, 0.1)

    toy_posterior = compute_log_posterior(thetas, true_phi, real_data, prior)

    print("toy_posterior generated")
    best_phi = optimize_phi(toy_posterior, thetas)
    return best_phi

best_phi = do_real_experiments()
print(best_phi)
