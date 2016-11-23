import numpy as np
# Use following line when running on HPC:
#iimport matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

def safe_ln(x, minval=0.0000000001):
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

    log_posterior = log_likelihoods + log_prior

    max_log_likelihood = max(log_likelihoods)
    log_likelihoods -= max_log_likelihood
    likelihood = np.exp(log_likelihoods)
    likelihood = likelihood / np.sum(likelihood)

    max_log_prior = max(log_prior)
    log_prior -= max_log_prior
    prior = np.exp(log_prior)
    prior = prior / np.sum(prior)

    product = prior * likelihood
    posterior =  product / np.sum(product)
    log_posterior = np.log(posterior)

    plt.plot(thetas, likelihood)
    best_like_theta = thetas[np.argmax(log_likelihoods)]
    title_string = (
        "P(X|theta,phi=%0.2f), max at %0.2f, run_iter: %s, phi_iter: %s, exp_iter: %s" %
        (phi, best_like_theta, str(run_iter), str(phi_iter), str(exp_iter)))
    plt.title(title_string)
    plt.xlabel("Thetas")
    plt.ylabel("Likelihood")
    fig_name = "plots/%s/%s/LL - Iteration, phi_iter: %s" %(str(run_iter),str(exp_iter),str(phi_iter))
    plt.savefig(str(fig_name))
    plt.clf()

    plt.plot(thetas, posterior)
    best_pos_theta = thetas[np.argmax(log_posterior)]
    title_string = (
        "P(theta | X, phi=%0.2f), max at %0.2f,run_iter: %s,exp_iter: %s" %
        (phi, best_pos_theta,str(run_iter),str(exp_iter)))
    plt.title(title_string)
    plt.xlabel("Thetas")
    plt.ylabel("Posterior")
    fig_name = "plots/%s/%s/LP - Iteration, phi iter: %s" %(str(run_iter),str(exp_iter),str(phi_iter))
    plt.savefig(str(fig_name))
    plt.clf()

    return log_posterior


N_experiments = 5

# plausible experimental settings.
phis = np.linspace(0, 3.14, 10) #np.array([0.09, 0.1, 0.11])

# plausible parameter range.
thetas = np.linspace(-3, 3, 1000)

# Initialize a uniform prior on theta, a plausible theta true and phi value.
log_prior = safe_ln(np.ones_like(thetas) / thetas.shape[0])
phi_real = 0.1
theta_true = 1.0

# run till convergence:
for i in range(10):
    # Generate data for the MAP estimate of theta.
    real_data = black_box(100, theta_true, phi_real, i)
    log_posterior = compute_log_posterior(thetas, phi_real, real_data, log_prior,i)
    log_prior = np.copy(log_posterior)

    # XXX: There seems to be some floating-point issues here. Is there
    # anything that we can do about it?
    posterior = np.exp(log_posterior)
    best_entropy = -np.sum(log_posterior * posterior)
    print(best_entropy)
    
    theta_map = thetas[np.argmax(log_posterior)]                   

    # alternative to argmax (instead we draw thetas from the posterior dist.)
    # i.e., use theta_drawn_from_posterior instead of theta_map at line #172 (?)
    posterior = posterior / np.sum(posterior)
    idx_of_theta_drawn_from_posterior = np.random.choice(range(thetas.shape[0]), size=1, p=posterior)
    theta_drawn_from_posterior = thetas[idx_of_theta_drawn_from_posterior][0]

    phi_eigs = []
    #phi_exp_log_posteriors = np.zeros((len(phis), len(thetas)))

    for phi_ind, phi in enumerate(phis):

        curr_eig = 0.0
        curr_log_posterior = []
        # These experiments are to average out randomness in computing the
        # information gain.
        for n in range(N_experiments):

            # Compute p(theta | D_fake, phi)
            toy_data = black_box(100, theta_map, phi)
            log_posterior_phi = compute_log_posterior(
                thetas, phi, toy_data, log_prior,i,phi_ind,n)
            curr_log_posterior.append(log_posterior_phi)
            curr_entropy = -np.sum(log_posterior_phi * np.exp(log_posterior_phi))
            curr_eig += best_entropy - curr_entropy

        #phi_exp_log_posteriors[phi_ind] = np.mean(curr_log_posterior, axis=0)
        phi_eigs.append(curr_eig / N_experiments)

    # Update phi and log-prior with the the best value of phi and the
    # log posterior.
    best_eig_ind = np.argmax(phi_eigs)
    phi_real = phis[best_eig_ind]

    title_string = ("EIG(phi), max at %0.2f, run_iter: %s" %(best_eig_ind, i))
    plt.title(title_string)
    plt.xlabel("phi")
    plt.ylabel("avg(EIG)")
    fig_name = "plots/%s/EIG_average" %(str(i))
    plt.savefig(str(fig_name))
    plt.clf()


