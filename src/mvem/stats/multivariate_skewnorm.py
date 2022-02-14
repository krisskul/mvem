import numpy as np
from scipy.stats import (multivariate_normal as mvn, norm)
from scipy.stats._multivariate import _squeeze_output
import scipy

def _check_params(mu, sigma, lmbda):
    """
    Internal method to check if passed in parameters are valid.

    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: Tuple of the parameters as np.ndarrays, if no assertion error
        occurs.
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    lmbda = np.asarray(lmbda)

    # reshape mu, lmbda to (d,1)
    if len(mu.shape) == 1: mu = mu[:, None]
    if len(lmbda.shape) == 1: lmbda = lmbda[:, None]
    
    # assert shapes (d,1), (d,d), (d,1)
    d00, d01 = mu.shape
    d10, d11 = lmbda.shape
    d20, d21 = sigma.shape
    assert d00==d10==d20, "mu, sigma, lmbda must have same dimension d"
    assert d20==d21, "sigma must be qudratic with shape dxd"
    assert d01==1 and d11==1, "second dimension of mu, lmbda must be 1"
    
    return mu, sigma, lmbda

def mean(mu, sigma, lmbda):
    """
    Mean function of the multivariate skew normal distribution.

    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The mean of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    mu, sigma, lmbda = _check_params(mu, sigma, lmbda)    

    val, vec = np.linalg.eig(sigma)
    sigma_half = vec @ np.diag(val**(1/2)) @ vec.T
    delta = lmbda / np.sqrt(1 + np.sum(lmbda**2))
    return mu + np.sqrt(2/np.pi) * sigma_half @ delta

def var(mu, sigma, lmbda):
    """
    Variance function of the multivariate skew normal distribution.

    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    mu, sigma, lmbda = _check_params(mu, sigma, lmbda)
    
    val, vec = np.linalg.eig(sigma)
    sigma_half = vec @ np.diag(val**(1/2)) @ vec.T
    delta = lmbda / np.sqrt(1 + np.sum(lmbda**2))
    return sigma - (2/np.pi) * sigma_half @ delta @ delta.T @ sigma_half

def logpdf(x, mu, sigma, lmbda):
    """
    Log-probability density function of the multivariate skew normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The log-density at each observation.
    :rtype: np.ndarray with shape (p,).
    """
    mu, sigma, lmbda = _check_params(mu, sigma, lmbda)
    
    x    = mvn._process_quantiles(x, len(mu))
    pdf  = mvn(mu.flatten(), sigma).logpdf(x)
    val, vec = np.linalg.eig(sigma)
    sigma_inv_half = vec @ np.diag(val**(-1/2)) @ vec.T
    a = (lmbda.T @ sigma_inv_half).flatten()
    cdf  = norm(0, 1).logcdf((x-mu.flatten()) @ a)
    return _squeeze_output(np.log(2) + pdf + cdf)

def pdf(x, mu, sigma, lmbda):
    """
    Probability density function of the multivariate skew normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The density at each observation.
    :rtype: np.ndarray with shape (p,).
    """
    return np.exp(logpdf(x, mu, sigma, lmbda))

def loglike(x, mu, sigma, lmbda):
    """
    Log-likelihood function of the multivariate skew normal distribution.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :return: The log-likelihood for given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, mu, sigma, lmbda))

def rvs(mu, sigma, lmbda, size=1):
    """
    Random number generator of the multivariate skew normal distribution.

    :param mu: The location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: The scale parameter. A positive semi-definite array with
        shape (p, p).
    :type sigma: np.ndarray
    :param lmbda: The skewness parameter with shape (p,).
    :type lmbda: np.ndarray
    :param size: The number of samples to draw. Defaults to 1.
    :type size: int, optional
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).
    """
    mu, sigma, lmbda = _check_params(mu, sigma, lmbda)

    # switch to (Omega, alpha) parameterization
    cov = sigma
    val, vec = np.linalg.eig(cov)
    cov_half = vec @ np.diag(val**(-1/2)) @ vec.T
    shape = cov_half @ lmbda

    aCa      = shape.T @ cov @ shape
    delta    = (1 / np.sqrt(1 + aCa)) * cov @ shape
    cov_star = np.block([[np.ones(1),     delta.T],
                        [delta, cov]])
    x        = mvn(np.zeros(len(shape)+1), cov_star).rvs(size)
    if size == 1: x = x[None, :]
    x0, x1   = x[:, 0], x[:, 1:]
    inds     = x0 <= 0
    x1[inds] = -1 * x1[inds]
    return x1 + mu.T


def fit(x, maxiter = 100, ptol = 1e-6, ftol = 1e-10, eps=0.9, return_loglike = False):
    """
    Estimate the parameters of the multivariate skew normal distribution
    using an EM algorithm.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param maxiter: The maximum number of iterations to use in the EM algorithm.
        Defaults to 100.
    :type maxiter: int, optional
    :param ptol: The relative convergence criterion for the estimated 
        parameters. Defaults to 1e-6.
    :type ptol: float, optional
    :param ftol: The relative convergence criterion for the log-likelihood 
        function. Defaults to np.inf.
    :type ftol: float, optional
    :param eps: Initialization shrinkage of delta parameter. Defaults to 0.9.
    :type eps: float, optional
    :param return_loglike: Return a list of log-likelihood values at each iteration. 
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<array> mu, <array> scale, <array> nu). Also returns
        a list of log-likelihood values at each iteration of the EM algorithm if 
        ``return_loglike=True``.
    :rtype: tuple
    """

    def _ll(xi, n, omega_inv_half, delta):
    
        d = xi.shape[1]
        lmbda = delta / np.sqrt(1 - np.sum(delta**2))
        wx = xi @ omega_inv_half
        cdf = scipy.stats.norm.cdf(wx @ lmbda)
        pdf = scipy.stats.multivariate_normal.pdf(wx, np.zeros(d), np.eye(d))
        ww = n * np.log(2) + np.sum(np.log(cdf)) + np.sum(np.log(pdf)) + n*np.log(np.linalg.det(omega_inv_half))
        return ww
    
    itereps = 10e-6
    n, p = x.shape
    
    # get pseudo moment estiators???
    m1 = np.mean(x, axis=0)
    center = x - m1
    m3 = np.mean(center**3, axis=0)
    a1 = np.sqrt(2/np.pi)
    b1 = (4/np.pi-1)*a1
    gam = np.sign(m3) * (np.abs(m3)/b1)**(1/3)

    mu0 = m1 - a1 * gam
    omega0 = (center.T @ center) / n + a1 * (gam[:, None] @ gam[None, :])

    val, vec = np.linalg.eig(omega0)
    omega0_inv_half = vec @ np.diag(val**(-1/2)) @ vec.T
    omega0_half = vec @ np.diag(val**(1/2)) @ vec.T

    delta0 = omega0_inv_half @ gam
    
    
    if (np.sum(delta0**2) > 1):
        delta0 = eps * delta0 / np.sqrt(np.sum(delta0**2))

    tau0 = 1
    mu1 = mu0
    omega1 = omega0
    delta1 = delta0
    
    ll_val = np.array([_ll(x - mu0, n, omega0_inv_half, delta0)])
    
    for i in range(maxiter):
        
        # E-step
        xi0 = x - mu0
        lmbda0 = delta0 / np.sqrt(1 - np.sum(delta0**2))
        w1 = 1 / np.sqrt(1 + np.sum(lmbda0**2))
        wa = xi0 @ (omega0_inv_half @ lmbda0)
        pdf = scipy.stats.norm.pdf(wa)
        cdf = scipy.stats.norm.cdf(wa)

        yexabs = tau0 * w1 * (pdf/cdf + wa)
        yex2   = tau0**2 * w1**2 * wa * (pdf/cdf + wa + 1/wa)
        
        # M-step
        mu1 = m1 - (omega0_half @ delta0)/tau0 * np.mean(yexabs)
        tau1 = np.mean(yex2)**(1/2)
        omega1 = (x-mu1).T @ (x-mu1) / n
        val, vec = np.linalg.eig(omega1)
        omega1_inv_half = vec @ np.diag(val**(-1/2)) @ vec.T
        omega1_half = vec @ np.diag(val**(1/2)) @ vec.T        
        delta1 = omega1_inv_half @ (yexabs @ (x-mu1))/(n*tau1)
        

        # measure log likelihood
        ll_val = np.append(ll_val, _ll(x-mu1, n, omega1_inv_half, delta1))

        # check for converagence (stopping criterion)
        #have_params_converged = \
        #    np.all(np.abs(mu1 - mu0) <= .5 * ptol * (abs(mu1) + abs(mu0))) & \
        #    np.all(np.abs(omega1 - omega0) <= .5 * ptol * (abs(omega1) + abs(omega0))) & \
        #    np.all(np.abs(delta1 - delta0) <= .5 * ptol * (abs(delta1) + abs(delta0)))
        has_fun_converged = np.abs(ll_val[-1] - ll_val[-2]) <= .5 * ftol * ( np.abs(ll_val[-1]) + np.abs(ll_val[-2]))
        if has_fun_converged: break

        # update old parameters
        mu0 = mu1
        omega0_inv_half = omega1_inv_half
        omega0_half = omega1_half
        delta0 = delta1
        tau0 = tau1

    #lmbda1 = delta0 / np.sqrt(1-np.sum(delta0**2))
    #val, vec = np.linalg.eig(omega1)
    #inv_half = vec @ np.diag(val**(-1)) @ vec.T
    #lmbda0 = inv_half @ delta1 / np.sqrt(1 - np.sum(delta1**2))
    lmbda0 = delta1 / np.sqrt(1 - np.sum(delta1**2))
    if return_loglike:
        return mu1, omega1, lmbda0, ll_val
    return mu1, omega1, lmbda0