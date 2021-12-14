import numpy as np
from scipy.stats import multivariate_t
from scipy.special import digamma, gammaln
import scipy

def pdf(x, loc, shape, df, allow_singular=False):
    return multivariate_t.pdf(x, loc, shape, df, allow_singular)
        
def logpdf(x, loc, shape, df, allow_singular=False):
    return multivariate_t.logpdf(x, loc, shape, df, allow_singular)

def rvs(loc, shape, df, size=1, random_state=None):
    return multivariate_t.rvs(loc, shape, df, size, random_state)

def mean(loc, shape, df):
    assert df > 1, "mean of mv student's t is not defined for df <= 1"
    return loc

def var(loc, shape, df):
    assert df > 2, "variance of mv student's t is not defined for df <= 2"
    return (df / (df - 2)) * shape

def fit(X, maxiter = 100, ptol = 1e-6, ftol = 1e-8):
    # EM Student's t:

    N, D = X.shape
    assert N > D, "Must have more observations than dimensions in _x_"
    
    # initialise values:
    nu = 4
    mu = np.mean(X, axis = 0)
    sigma = np.linalg.inv((nu / (nu - 2)) * np.cov(X.T))
    log_likelihood = np.array([np.sum(multivariate_t.logpdf(X, mu, np.linalg.inv(sigma), nu))])

    for i in range(maxiter):
        
        # save old values
        nu_old = nu
        mu_old = mu
        sigma_old = sigma
        
        # Expectation Step
        X_ = X - mu
        delta2 = np.sum((X_ @ sigma) * X_, axis=1)
        E_tau = (nu + D) / (nu + delta2)
        E_logtau = digamma(0.5 * (nu + D)) - np.log(0.5 * (nu + delta2))

        # Maximization Step
        mu = (E_tau @ X) / np.sum(E_tau)
        alpha = X - mu
        sigma = np.linalg.inv((alpha * E_tau[:, np.newaxis]).T @ alpha / N)

        # ... if method = "EM"
        func = lambda x: digamma(x/2.) - np.log(x/2.) - 1 - np.mean(E_logtau) + np.mean(E_tau)
        nu = scipy.optimize.fsolve(func, nu) # set min, max ???
                    
        # check for converagence (stopping criterion)
        have_params_converged = \
            np.all(np.abs(mu - mu_old) <= .5 * ptol * (abs(mu) + abs(mu_old))) & \
            np.all(np.abs(nu - nu_old) <= .5 * ptol * (abs(nu) + abs(nu_old))) & \
            np.all(np.abs(sigma - sigma_old) <= .5 * ptol * (abs(sigma) + abs(sigma_old)))
        if ftol < np.inf:
            ll = np.sum(multivariate_t.logpdf(X, mu, np.linalg.inv(sigma), nu))
            has_fun_converged = np.abs(ll - log_likelihood[-1]) <= .5 * ftol * (
                np.abs(ll) + np.abs(log_likelihood[-1]))
            log_likelihood = np.append(log_likelihood, ll)
        else:
            has_fun_converged = True
        if have_params_converged and has_fun_converged: break

    sigma = np.linalg.inv(sigma)    
    return mu, sigma, nu, log_likelihood