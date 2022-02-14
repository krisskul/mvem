import numpy as np
from mvem.stats import multivariate_genhyperbolic as ghypmv

def logpdf(x, chi, psi, mu, sigma, gamma):
    """
    Log-probability density function of the multivariate normal-inverse Gaussian
    distribution. We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param chi: Univariate parameter > 0.
    :type chi: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The log-density at each observation.
    :rtype: np.ndarray with shape (n,).
    """
    return ghypmv.logpdf(x, -0.5, chi, psi, mu, sigma, gamma)

def pdf(x, chi, psi, mu, sigma, gamma):
    """
    Probability density function of the multivariate normal-inverse Gaussian
    distribution. We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param chi: Univariate parameter > 0.
    :type chi: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The density at each observation.
    :rtype: np.ndarray with shape (n,).
    """
    return ghypmv.pdf(x, -0.5, chi, psi, mu, sigma, gamma)

def loglike(x, chi, psi, mu, sigma, gamma):
    """
    Log-likelihood function of the multivariate normal-inverse Gaussian
    distribution. We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param chi: Univariate parameter > 0.
    :type chi: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The log-likelihood given all observations and parameters.
    :rtype: float
    """
    return np.sum(logpdf(x, chi, psi, mu, sigma, gamma))

def rvs(chi, psi, mu, sigma, gamma, size):
    """
    Random number generator of the multivariate normal-inverse Gaussian
    distribution. We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param chi: Univariate parameter > 0.
    :type chi: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :param size: The number of samples to draw. Defaults to 1.
    :type size: int, optional
    :return: The random p-variate numbers generated.
    :rtype: np.ndarray with shape (n, p).
    """
    return ghypmv.rvs(-0.5, chi, psi, mu, sigma, gamma, size)

def mean(chi, psi, mu, sigma, gamma):
    """
    Mean function of the multivariate normal-inverse Gaussian
    distribution. We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param chi: Univariate parameter > 0.
    :type chi: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The mean of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    return ghypmv.mean(-0.5, chi, psi, mu, sigma, gamma)

def var(chi, psi, mu, sigma, gamma):
    """
    Variance function of the multivariate normal-inverse Gaussian
    distribution. We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param chi: Univariate parameter > 0.
    :type chi: float
    :param psi: Univariate parameter > 0.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (n,).
    """
    return ghypmv.var(-0.5, chi, psi, mu, sigma, gamma)

def fit(x, alpha_bar=1, symmetric=False, standardize=False, nit=2000, reltol=1e-8,
        abstol=1e-7, silent=False, fmu=None, fsigma=None, fgamma=None, return_loglike=False):
    """
    Estimate the parameters of the normal-inverse gaussian distribution. We
    use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param alpha_bar: The initial value of alpha_bar, a positive real number, 
        where alpha_bar = chi =  psi.
    :type alpha_bar: float, optional
    :param symmetric: Whether to fit a symmetric distribution or not. Default 
        to False.
    :type symmetric: bool, optional
    :param standardize: Whether to standardize the data before fitting or not.
        Default to False.
    :type standardize: bool, optional
    :param nit: The maximum number of iterations to use in the EM algorithm.
        Defaults to 2000.
    :type nit: int, optional
    :param reltol: The relative convergence criterion for the log-likelihood 
        function. Defaults to 1e-8.
    :type reltol: float, optional
    :param abstol: The relative convergence criterion for the log-likelihood 
        function. Defaults to 1e-7.
    :type abstol: float, optional
    :param silent: Whether to print the log-likelihoods and parameter estimates
        during fitting or not. Defaults to False.
    :type silent: bool, optional
    :param fmu: If fmu!=None, force mu to fmu. Defaults to None.
    :type fmu: np.ndarray, optional
    :param fsigma: If fsigma!=None, force sigma to fsigma. Defaults to None.
    :type fsigma: np.ndarray, optional
    :param fgamma: If fgamma!=None, force gamma to fgamma. Defaults to None.
    :type fgamma: np.ndarray, optional
    :param return_loglike: Return a list of log-likelihood values at each iteration. 
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<float> chi, <float> psi, <array> mu, <array> sigma,
        <array> gamma). Also returns a list of log-likelihood values at each iteration
        of the EM algorithm if ``return_loglike=True``.
    :rtype: tuple
    """

    opt_pars = {"lmbda": False, "alpha_bar": True, "mu": fmu is None,
                "sigma": fsigma is None, "gamma": fgamma is None}
    
    fit = ghypmv.fitghypmv(
        x, lmbda=-0.5, alpha_bar=alpha_bar, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars)
    
    chi, psi = fit["alpha_bar"], fit["alpha_bar"]
    if return_loglike:
        return chi, psi, fit["mu"], fit["sigma"], fit["gamma"], fit["ll"]
    return chi, psi, fit["mu"], fit["sigma"], fit["gamma"]


