import numpy as np
from . import multivariate_genhyperbolic as ghypmv

def logpdf(x, lmbda, chi, mu, sigma, gamma):
    """
    Log-probability density function of the generalised hyperbolic skew t-distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    lmbda: float
        lmbda < 0
    chi: float
        chi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    return ghypmv.logpdf(x, lmbda, chi, 0, mu, sigma, gamma)

def pdf(x, lmbda, chi, mu, sigma, gamma):
    """
    Probability density function of the generalised hyperbolic skew t-distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    lmbda: float
        lmbda < 0
    chi: float
        chi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    return ghypmv.pdf(x, lmbda, chi, 0, mu, sigma, gamma)

def loglike(x, lmbda, chi, mu, sigma, gamma):
    """
    Log-likelihood function of the generalised hyperbolic skew t-distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    lmbda: float
        lmbda < 0
    chi: float
        chi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    return np.sum(logpdf(x, lmbda, chi, mu, sigma, gamma))

def rvs(lmbda, chi, mu, sigma, gamma, size):
    """
    Random number generator of the generalised hyperbolic skew t-distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    lmbda: float
        lmbda < 0
    chi: float
        chi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    size: int
        number of samples to draw
    """
    return ghypmv.rvs(lmbda, chi, 0, mu, sigma, gamma, size)

def mean(lmbda, chi, mu, sigma, gamma):
    """
    Mean function of the generalised hyperbolic skew t-distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    lmbda: float
        lmbda < 0
    chi: float
        chi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    return ghypmv.mean(lmbda, chi, 0, mu, sigma, gamma)

def var(lmbda, chi, mu, sigma, gamma):
    """
    Variance function of the generalised hyperbolic skew t-distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    lmbda: float
        lmbda < 0
    chi: float
        chi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    return ghypmv.var(lmbda, chi, 0, mu, sigma, gamma)

def fit(x, lmbda=-1, symmetric=False, standardize=False, nit=2000, reltol=1e-8,
        abstol=1e-7, silent=False, fmu=None, fsigma=None, fgamma=None, return_loglike=False):
    """
    Estimate the parameters of the generalised hyperbolic skew t-distribution. We
    use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    lmbda: float
        initial value of lmbda
    symmetric: boolean, optional
        fit a symmetric distribution (default=False)
    standardize: boolean, optional
        standardize the data before fitting (default=False)
    nit: int
        maximum number of iterations in EM algorithm (default=2000)
    reltol: float
        relative convergence criterion for log-likelihood (default=1e-8)
    abstol: float
        absolute convergence criterion for log-likelihood (default=1e-7)
    silent: boolean
        print likelihoods during fitting (default=False)
    fmu: float or None
        if fmu!=None, force mu to fmu (default=None)
    fsigma: float or None
        if fsigma!=None, force sigma to fsigma (default=None)
    fgamma: float or None
        if fgamma!=None, force gamma to fgamma (default=None)
    return_loglike: boolean
        return log-likelihood values (default=False)

    Returns
    -------
    lmbda: float
    chi: float
    mu: np.ndarray
    sigma: np.ndarray
    gamma: np.ndarray
    log_likelihoods: list (returned if return_loglike = True)
    """

    opt_pars = {"lmbda": True, "alpha_bar": False, "mu": fmu is None,
                "sigma": fsigma is None, "gamma": fgamma is None}
    
    fit = ghypmv.fitghypmv(
        x, lmbda=lmbda, alpha_bar=0, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars)
    
    chi = -2 * (fit["lmbda"] + 1)
    if return_loglike:
        return fit["lmbda"], chi, fit["mu"], fit["sigma"], fit["gamma"], fit["ll"]
    return fit["lmbda"], chi, fit["mu"], fit["sigma"], fit["gamma"]


