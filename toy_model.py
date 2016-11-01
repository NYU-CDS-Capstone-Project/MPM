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
    Gives likelihood of P(X | theta, phi)
    """
    # Generate samples to estimate the empirical distribution.
    samples = black_box(10**6, theta, phi)
    n, bins, _ = plt.hist(samples, 1000, normed=True)
    bin_indices = np.searchsorted(bins, X) - 1

    # Clip values outside the interval.
    bin_indices[bin_indices == -1] = 0
    bin_indices[bin_indices == len(n)] = len(n) - 1
    n_counts = n[bin_indices]
    P_X_given_theta = n_counts / np.sum(n_counts)
    return np.sum(safe_ln(P_X_given_theta))


def compute_log_posterior(thetas, phi, X, log_prior, toy_iter="init", phi_iter="init"):
    """
    Compute P(theta | phi, X)
    """
    log_posterior = np.empty_like(thetas)
    log_likelihoods = np.empty_like(thetas)

    
    for i, theta in enumerate(thetas):
        # Find log(\prod_{i=1}^n P(X_i | t, phi)
        log_like = log_likelihood(X, theta, phi)
        log_likelihoods[i] = log_like
    
    max_log_like = max(log_likelihoods)
    log_likelihoods -= max_log_like
    log_posterior = log_likelihoods + log_prior

    posterior = np.exp(log_posterior)
    sum_posterior = np.sum(posterior)
    norm_posterior = posterior / sum_posterior
    log_posterior = np.log(norm_posterior)

    plt.plot(thetas, log_likelihoods)
    best_like_theta = thetas[np.argmax(log_likelihoods)]
    title_string = (
        "log_likelihood P(X | theta, phi=%0.2f), max at %0.2f, %s" %
        (phi, best_like_theta, str(toy_iter)))
    plt.title(title_string)
    plt.xlabel("Thetas")
    plt.ylabel("Log Likelihood")
    plt.savefig("LL - Iteration, phi iter: %s, toy iter: %s" %
                (str(phi_iter), str(toy_iter)))
    plt.clf()

    plt.plot(thetas, log_posterior)
    best_pos_theta = thetas[np.argmax(log_posterior)]
    title_string = (
        "log_posterior P(theta | X, phi=%0.2f), max at %0.2f, %s" %
        (phi, best_pos_theta, str(toy_iter)))
    plt.title(title_string)
    plt.xlabel("Thetas")
    plt.ylabel("Log Posterior") 
    plt.savefig("LP - Iteration, phi iter: %s, toy iter: %s" %
                (str(phi_iter), str(toy_iter)))
    plt.clf()

    return np.array(log_posterior)


def optimize_phi(log_posterior, log_prior, thetas):
    # entropy of posterior
    best_entropy = np.sum(log_posterior)

    phis = np.linspace(0.5, 1.5, 20)
    mean_inf_gains = []
    best_phys = []
    N_toys = 10
    for phi_iter, phi in enumerate(phis):
        print("Phi Iter: %d" % phi_iter)
        inf_gain = []

        for i in range(N_toys):
            print("optimize_phi " + str(i))
            theta_best = np.argmax(log_posterior)
            toy_data = black_box(1000, theta_best, phi, i)
            log_posterior = compute_log_posterior(
                thetas, phi, toy_data, log_prior, phi_iter, i)
            curr_entropy = np.sum(log_posterior)
            inf_gain.append(best_entropy - curr_entropy)
            best_phys.append(phis[np.argmax(inf_gain)])
        inf_gains.append(np.mean(inf_gain))
    return phis[np.argmax(inf_gains)]


def do_real_experiments(phi, theta, log_prior, thetas):
    """
    Estimate the true value of phi for an experiment.
    """

    # Generate 1000 samples from the black box.
    real_data =  black_box(1000, theta, phi, 0)

    # Prior:
    # Assume P(theta | phi) is a gaussian with mean 1.0
    # and std 0.1

    log_posterior = compute_log_posterior(thetas, phi, real_data, log_prior)

    print("toy_posterior generated")
    best_phi = optimize_phi(log_posterior, log_prior, thetas)
    return best_phi, log_posterior

N_experiments = 5
log_prior = np.log(norm.pdf(thetas, 1.0, 0.1))
phi = 0.1
theta = 1.0
# generate a set of plausible thetas
thetas = np.linspace(0.5, 1.5, 100)

for i in range(N_experiments):
    phi, log_posterior = do_real_experiments(phi, theta, log_prior, thetas)
    theta = np.argmax(log_posterior)
