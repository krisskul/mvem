from scipy.stats import multivariate_normal
import numpy as np

def fit(X, return_loglike=False):
    """
    Estimate the parameters of the multivariate normal distribution using
    maximum likelihood estimates.

    Parameters
    ----------
    X: np.ndarray
        (n, p) array of n p-variate observations
    return_loglike: boolean
        whether to return the log-likelihood
    """
    mean = np.mean(X, axis = 0)
    cov = np.cov(X.T)
    if return_loglike:
        return mean, cov, loglike(X, mean, cov)
    return mean, cov

def pdf(x, mean, cov, allow_singular=False):
    """
    Probability density function of the multivariate normal distribution.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    mean: np.ndarray or list
        (p,) array of location parameters
    cov: np.ndarray or list
        (p, p) positive semi-definite array of scale parameters
    allow_singular: boolean
        whether to allow a singular matrix
    """
    return multivariate_normal.pdf(x, mean, cov, allow_singular)

def logpdf(x, mean, cov, allow_singular=False):
    """
    Log-probability density function of the multivariate normal distribution.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    mean: np.ndarray or list
        (p,) array of location parameters
    cov: np.ndarray or list
        (p, p) positive semi-definite array of scale parameters
    allow_singular: boolean
        whether to allow a singular matrix
    """
    return multivariate_normal.logpdf(x, mean, cov, allow_singular)

def loglike(x, mean, cov, allow_singular=False):
    """
    Log-likelihood function of the multivariate normal distribution.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    mean: np.ndarray or list
        (p,) array of location parameters
    cov: np.ndarray or list
        (p, p) positive semi-definite array of scale parameters
    allow_singular: boolean
        whether to allow a singular matrix
    """
    return np.sum(logpdf(x, mean, cov, allow_singular))

def cdf(x, mean, cov, allow_singular=False, maxpts=1000000, abseps=1e-5, releps=1e-5):
    """
    Cumulative density function of the multivariate normal distribution.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    mean: np.ndarray or list
        (p,) array of location parameters
    cov: np.ndarray or list
        (p, p) positive semi-definite array of scale parameters
    allow_singular: boolean
        whether to allow a singular matrix
    maxpts: integer, optional
        the maximum number of points to use for integration
    abseps: float, optional
        absolute error tolerance (default 1e-5)
    releps: float, optional
        relative error tolerance (default 1e-5)
    """
    maxpts = maxpts * x.shape[1]
    return multivariate_normal.cdf(x, mean, cov, allow_singular, maxpts, abseps, releps)

def logcdf(x, mean, cov, allow_singular=False, maxpts=1000000, abseps=1e-5, releps=1e-5):
    """
    Log-cumulative density function of the multivariate normal distribution.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    mean: np.ndarray or list
        (p,) array of location parameters
    cov: np.ndarray or list
        (p, p) positive semi-definite array of scale parameters
    allow_singular: boolean
        whether to allow a singular matrix
    maxpts: integer, optional
        the maximum number of points to use for integration
    abseps: float, optional
        absolute error tolerance (default 1e-5)
    releps: float, optional
        relative error tolerance (default 1e-5)
    """
    maxpts = maxpts * x.shape[1]
    return multivariate_normal.logcdf(x, mean, cov, allow_singular, maxpts, abseps, releps)

def rvs(mean, cov, size=1, random_state=None):
    """
    Random number generator of the multivariate normal distribution.

    Parameters
    ----------
    mean: np.ndarray or list
        (p,) array of location parameters
    cov: np.ndarray or list
        (p, p) positive semi-definite array of scale parameters
    random_state : {None, int, np.random.RandomState, np.random.Generator}, optional
        used for drawing random variates.
    """
    return multivariate_normal.rvs(mean, cov, size, random_state)

def mean(mean, cov):
    """
    Mean function of the multivariate normal distribution.

    Parameters
    ----------
    mean: np.ndarray or list
        (p,) array of location parameters
    cov: np.ndarray or list
        (p, p) positive semi-definite array of scale parameters
    """
    return mean

def var(mean, cov):
    """
    Variance function of the multivariate normal distribution.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    mean: np.ndarray or list
        (p,) array of location parameters
    cov: np.ndarray or list
        (p, p) positive semi-definite array of scale parameters
    """
    return cov