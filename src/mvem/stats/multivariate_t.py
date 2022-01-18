import numpy as np
import scipy.stats
from scipy.special import digamma, gammaln
import scipy

def pdf(x, loc, shape, df, allow_singular=False):
    """
    Probability density function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    loc: np.ndarray or list
        (p,) array o location parameters
    shape: np.ndarray or list
        (p, p) positive semi-definite array of shape parameters
    df: float
        positive scalar giving shape parameter
    allow_singular: boolean
        whether to allow a singular matrix
    """
    return scipy.stats.multivariate_t.pdf(x, loc, shape, df, allow_singular)
        
def logpdf(x, loc, shape, df, allow_singular=False):
    """
    Log-probability density function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    loc: np.ndarray or list
        (p,) array o location parameters
    shape: np.ndarray or list
        (p, p) positive semi-definite array of shape parameters
    df: float
        positive scalar giving shape parameter
    allow_singular: boolean
        whether to allow a singular matrix
    """
    return scipy.stats.multivariate_t.logpdf(x, loc, shape, df, allow_singular)

def loglike(x, loc, shape, df):
    """
    Log-likelihood function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    loc: np.ndarray or list
        (p,) array o location parameters
    shape: np.ndarray or list
        (p, p) positive semi-definite array of shape parameters
    df: float
        positive scalar giving shape parameter
    """
    return np.sum(logpdf(x, loc, shape, df))

def rvs(loc, shape, df, size=1, random_state=None):
    """
    Random number generator of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    Parameters
    ----------
    loc: np.ndarray or list
        (p,) array o location parameters
    shape: np.ndarray or list
        (p, p) positive semi-definite array of shape parameters
    df: float
        positive scalar giving shape parameter
    size: int
        the number of random numbers to draw
    random_state : {None, int, np.random.RandomState, np.random.Generator}, optional
        used for drawing random variates.
    """
    return scipy.stats.multivariate_t.rvs(loc, shape, df, size, random_state)

def mean(loc, shape, df):
    """
    Mean of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    Parameters
    ----------
    loc: np.ndarray or list
        (p,) array o location parameters
    shape: np.ndarray or list
        (p, p) positive semi-definite array of shape parameters
    df: float
        positive scalar giving shape parameter
    """
    assert df > 1, "mean of mv student's t is not defined for df <= 1"
    return loc

def var(loc, shape, df):
    """
    Variance of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    Parameters
    ----------
    loc: np.ndarray or list
        (p,) array o location parameters
    shape: np.ndarray or list
        (p, p) positive semi-definite array of shape parameters
    df: float
        positive scalar giving shape parameter
    """
    assert df > 2, "variance of mv student's t is not defined for df <= 2"
    return (df / (df - 2)) * shape

def fit(X, maxiter = 100, ptol = 1e-6, ftol = 1e-8, return_loglike = False):
    """
    Fit the parameters of the multivariate Student's t-distribution to data
    using an EM algorithm. We use the location-scale parameterisation.

    Parameters
    ----------
    X: np.ndarray
        (n, p) array of n p-variate observations
    maxiter: int
        max nunber of iterations in the EM algorithm
    ptol: float
        convergence tolerance for parameters
    ftol: float
        convergence tolerance for log-likelihood
    return_loglike: boolean
        whether to return the list of log-likelihoods

    Returns
    -------
    mu: np.ndarray
    scale: np.ndarray
    df: np.ndarray
    log_likelihoods: list (returned if return_loglike = True)
    """

    # EM Student's t:

    N, D = X.shape
    assert N > D, "Must have more observations than dimensions in _x_"
    
    # initialise values:
    nu = 4
    mu = np.mean(X, axis = 0)
    sigma = np.linalg.inv((nu / (nu - 2)) * np.cov(X.T))
    log_likelihood = np.array([np.sum(scipy.stats.multivariate_t.logpdf(X, mu, np.linalg.inv(sigma), nu))])

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
            ll = np.sum(scipy.stats.multivariate_t.logpdf(X, mu, np.linalg.inv(sigma), nu))
            has_fun_converged = np.abs(ll - log_likelihood[-1]) <= .5 * ftol * (
                np.abs(ll) + np.abs(log_likelihood[-1]))
            log_likelihood = np.append(log_likelihood, ll)
        else:
            has_fun_converged = True
        if have_params_converged and has_fun_converged: break

    sigma = np.linalg.inv(sigma)

    if return_loglike:
        return mu, sigma, nu, log_likelihood

    return mu, sigma, nu