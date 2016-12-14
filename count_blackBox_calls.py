import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from createDirStructure import mkdir_all 
rng = np.random.RandomState(0)
from sys import exit

def black_box(n_samples, theta=1.0, phi=0.2, random_state=None):
    return n_samples


def log_likelihood(X, theta, phi):
    nSample = black_box(10**6, theta, phi)
    return nSample


def compute_log_posterior(thetas, phi, X, log_prior, run_iter="init", phi_iter="init", exp_iter="init"):
    nCall = 0
    for i, theta in enumerate(thetas):
        # Find log(\prod_{i=1}^n P(X_i | t, phi)
        nSample = log_likelihood(X, theta, phi)
        nCall = nCall+1
        #"nCall X nSample ="+
    toatal = str(nSample)+"X"+str(nCall)
    return toatal

N_experiments = 20
phis = np.linspace(0, 2*np.pi, 10)
thetas = np.linspace(-3, 3, 200)
phi_real = 0.1
theta_true = 1.0
n_iter = 10
mkdir_all("plots",n_iter,N_experiments)
print("Start parameters:")
print("n_iter="+str(n_iter)) 
print("N_experiments="+str(N_experiments))
print("n_thetas="+str(len(thetas)))
print("n_phis="+str(len(phis)))
print("List of call to the blak box:")

for i in range(n_iter):
    call1 = black_box(100, theta_true, phi_real, i)
    print "nSamples X n_iter="+str(call1)+"X"+str(n_iter)
    callPost1 = compute_log_posterior(thetas, phi_real, "", "",i)
    print "nSamples X nThetas X n_iter="+callPost1+"X"+str(n_iter)
    for phi_ind, phi in enumerate(phis):
        for n in range(N_experiments):
            call2 = black_box(100, "", phi)
            print "nSamples X n_iter X N_experiments X n_phis="+str(call2)+"X"+str(n_iter)+"X"+str(N_experiments)+"X"+str(len(phis))
            callPost2 = compute_log_posterior(
                thetas, phi, "", "",i,phi_ind,n)
            print  "nSamples X nThetas X n_iter X N_experiments X n_phis="+callPost2+"X"+str(n_iter)+"X"+str(N_experiments)+"X"+str(len(phis))
            exit()

# Start parameters:
# n_iter=10
# N_experiments=20
# n_thetas=200
# n_phis=10
# List of call to the blak box:
# nSamples X n_iter=100X10
# nSamples X nThetas X n_iter=1000000X200X10
# nSamples X n_iter X N_experiments X n_phis=100X10X20X10
# nSamples X nThetas X n_iter X N_experiments X n_phis=1000000X200X10X20X10


