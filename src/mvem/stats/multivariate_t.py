import numpy as np
import scipy.stats
from scipy.special import digamma, gammaln
import scipy

def pdf(x, loc, shape, df, allow_singular=False):
    """
    Probability density function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :param allow_singular: Whether to allow a singular matrix.
    :type allow_singular: bool, optional.
    :return: The density at each observation.
    :rtype: np.ndarray with shape (p,).
    """
    return scipy.stats.multivariate_t.pdf(x, loc, shape, df, allow_singular)
        
def logpdf(x, loc, shape, df, allow_singular=False):
    """
    Log-probability density function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :param allow_singular: Whether to allow a singular matrix.
    :type allow_singular: bool, optional.
    :return: The log-density at each observation.
    :rtype: np.ndarray with shape (p,).
    """
    return scipy.stats.multivariate_t.logpdf(x, loc, shape, df, allow_singular)

def loglike(x, loc, shape, df, allow_singular=False):
    """
    Log-likelihood function of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :param allow_singular: Whether to allow a singular matrix.
    :type allow_singular: bool, optional.
    :return: The log-likelihood for given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, loc, shape, df, allow_singular))

def rvs(loc, shape, df, size=1, random_state=None):
    """
    Random number generator of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :param size: The number of samples to draw. Defaults to 1.
    :type size: int, optional
    :param random_state: Used for drawing random variates. Defaults to None.
    :type random_state: None, int, np.random.RandomState, np.random.Generator, optional
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).
    """
    return scipy.stats.multivariate_t.rvs(loc, shape, df, size, random_state)

def mean(loc, shape, df):
    """
    Mean of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :return: The mean of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    assert df > 1, "mean of mv student's t is not defined for df <= 1"
    return loc

def var(loc, shape, df):
    """
    Variance of the multivariate Student's t-distribution.
    We use the location-scale parameterisation.

    :param loc: The location parameter with shape (p,).
    :type loc: np.ndarray
    :param shape: The shape parameter. A positive semi-definite array with
        shape (p, p).
    :type shape: np.ndarray
    :param df: The degrees of freedom of the distribution, > 0.
    :type df: float
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    assert df > 2, "variance of mv student's t is not defined for df <= 2"
    return (df / (df - 2)) * shape

def fit(X, maxiter = 100, ptol = 1e-6, ftol = 1e-8, return_loglike = False):
    """
    Fit the parameters of the multivariate Student's t-distribution to data
    using an EM algorithm. We use the location-scale parameterisation.

    :param X: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type X: np.ndarray
    :param maxiter: The maximum number of iterations to use in the EM algorithm.
        Defaults to 100.
    :type nit: int, optional
    :param ptol: The relative convergence criterion for the estimated 
        parameters. Defaults to 1e-6.
    :type ptol: float, optional
    :param ftol: The relative convergence criterion for the log-likelihood 
        function. Defaults to np.inf.
    :type ftol: float, optional
    :param return_loglike: Return a list of log-likelihood values at each iteration. 
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<array> mu, <array> scale, <float> df). Also returns
        a list of log-likelihood values at each iteration of the EM algorithm if 
        ``return_loglike=True``.
    :rtype: tuple
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