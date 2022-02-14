from scipy.stats import multivariate_normal
import numpy as np

def fit(X, return_loglike=False):
    """
    Estimate the parameters of the multivariate normal distribution using
    maximum likelihood estimates.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param return_loglike: Return a list of log-likelihood values at each iteration. 
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<array> mu, <array> sigma). Also returns a list of 
        log-likelihood values at each iteration of the EM algorithm if ``return_loglike=True``.
    :rtype: tuple
    """
    mean = np.mean(X, axis = 0)
    cov = np.cov(X.T)
    if return_loglike:
        return mean, cov, loglike(X, mean, cov)
    return mean, cov

def pdf(x, mean, cov, allow_singular=False):
    """
    Probability density function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :return: The density at each observation.
    :rtype: np.ndarray with shape (n,).        
    """
    return multivariate_normal.pdf(x, mean, cov, allow_singular)

def logpdf(x, mean, cov, allow_singular=False):
    """
    Log-probability density function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :return: The log-density at each observation.
    :rtype: np.ndarray with shape (n,).        
    """
    return multivariate_normal.logpdf(x, mean, cov, allow_singular)

def loglike(x, mean, cov, allow_singular=False):
    """
    Log-likelihood function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :return: The log-likelihood for given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, mean, cov, allow_singular))

def cdf(x, mean, cov, allow_singular=False, maxpts=1000000, abseps=1e-5, releps=1e-5):
    """
    Cumulative density function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :param maxpts: The maximum number of points to use for integration.
    :type maxpts: int, optional
    :param abseps: The absolute error tolerance. Defaults to 1e-5.
    :type abseps: float, optional
    :param releps: The relative error tolerance. Defaults 1e-5.
    :type releps: float, optional
    :return: The cumulative density given all observations and parameters.
    :rtype: float
    """
    maxpts = maxpts * x.shape[1]
    return multivariate_normal.cdf(x, mean, cov, allow_singular, maxpts, abseps, releps)

def logcdf(x, mean, cov, allow_singular=False, maxpts=1000000, abseps=1e-5, releps=1e-5):
    """
    Log-cumulative density function of the multivariate normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param allow_singular: Whether to allow a singular matrix. Defaults to
        False.
    :type allow_singular: bool, optional
    :param maxpts: The maximum number of points to use for integration.
    :type maxpts: int, optional
    :param abseps: The absolute error tolerance. Defaults to 1e-5.
    :type abseps: float, optional
    :param releps: The relative error tolerance. Defaults 1e-5.
    :type releps: float, optional
    :return: The log-cumulative density given all observations and parameters.
    :rtype: float
    """
    maxpts = maxpts * x.shape[1]
    return multivariate_normal.logcdf(x, mean, cov, allow_singular, maxpts, abseps, releps)

def rvs(mean, cov, size=1, random_state=None):
    """
    Random number generator of the multivariate normal distribution.

    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :param size: The number of samples to draw. Defaults to 1.
    :type size: int, optional
    :param random_state: Used for drawing random variates. Defaults to None.
    :type random_state: None, int, np.random.RandomState, np.random.Generator, optional
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).       
    """
    return multivariate_normal.rvs(mean, cov, size, random_state)

def mean(mean, cov):
    """
    Mean function of the multivariate normal distribution.

    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :return: The mean of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    return mean

def var(mean, cov):
    """
    Variance function of the multivariate normal distribution.

    :param mean: The mean of the normal distribution. A parameter with
        shape (p,).
    :type mean: np.ndarray
    :param cov: The covariance of the normal distribution. A positive
        semi-definite array with shape (p, p).
    :type cov: np.ndarray
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    return cov