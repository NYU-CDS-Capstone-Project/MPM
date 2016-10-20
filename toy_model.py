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
    return np.sum(safe_ln(P_X_given_theta))

def compute_posterior(thetas, phi, X, prior):
    """
    Compute P(theta | phi, X)
    """
    log_posterior = []
    
    for i in range(len(thetas)):
        # Find log(\prod_{i=1}^n P(X_i | t, phi)
        log_like = log_likelihood(X, thetas[i], phi)
        log_prior = np.log(prior[i])
        log_posterior.append(log_like + log_prior)

    posterior = np.array(np.exp(log_posterior))
    normalize = np.sum(posterior)
    return posterior / normalize

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
            print "optimize_phi " + str(i)
            toy_data = black_box(1000, 1.0, phi, i)
            posterior = compute_posterior(thetas, phi, toy_data, posterior)
            curr_entropy = np.sum(posterior * np.log(posterior))
            inf_gain.append(best_entropy - curr_entropy)
            best_phys.append(phis[np.argmax(inf_gain)])
        inf_gains.append(np.mean(inf_gain))
    return phis[np.argmax(inf_gains)]


def do_real_experiments():
    # generate a set of plausible thetas
    phi = 0.1
    thetas = np.linspace(0.5, 1.5, 100)
    real_data =  black_box(1000, 1.0, phi, 0)
    # Prior:
    # Assume P(theta | phi) is a gaussian with mean 1.0
    # and std phi
    prior = [norm.pdf(theta, 1.0, 1.0) for theta in thetas]
    toy_posterior = compute_posterior(thetas, phi, real_data, prior)
    print "toy_posterior generated"
    best_phi = optimize_phi(toy_posterior, thetas)
    return best_phis

best_phis = do_real_experiments()
print best_phis