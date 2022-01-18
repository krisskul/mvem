import numpy as np
from . import multivariate_genhyperbolic as ghypmv

def logpdf(x, chi, psi, mu, sigma, gamma):
    """
    Log-probability density function of the multivariate hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    chi: float
        chi > 0
    psi: float
        psi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    lmbda = (len(mu)+1)/2
    return ghypmv.logpdf(x, lmbda, chi, psi, mu, sigma, gamma)

def pdf(x, chi, psi, mu, sigma, gamma):
    """
    Probability density function of the multivariate hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    chi: float
        chi > 0
    psi: float
        psi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    lmbda = (len(mu)+1)/2
    return ghypmv.pdf(x, lmbda, chi, psi, mu, sigma, gamma)

def loglike(x, chi, psi, mu, sigma, gamma):
    """
    Log-likelihood function of the multivariate hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    chi: float
        chi > 0
    psi: float
        psi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    return np.sum(logpdf(x, chi, psi, mu, sigma, gamma))

def rvs(chi, psi, mu, sigma, gamma, size):
    """
    Random number generator of the multivariate hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    chi: float
        chi > 0
    psi: float
        psi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    size: int
        number of samples to draw
    """
    lmbda = (len(mu)+1)/2
    return ghypmv.rvs(lmbda, chi, psi, mu, sigma, gamma, size)

def mean(chi, psi, mu, sigma, gamma):
    """
    Mean function of the multivariate hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    chi: float
        chi > 0
    psi: float
        psi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    lmbda = (len(mu)+1)/2
    return ghypmv.mean(lmbda, chi, psi, mu, sigma, gamma)

def var(chi, psi, mu, sigma, gamma):
    """
    Variance function of the multivariate hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    chi: float
        chi > 0
    psi: float
        psi > 0
    mu: np.ndarray or list
        (p,) array
    sigma: np.ndarray or list
        (p, p) positive semi-definite array
    gamma: np.ndarray or list
        (p,) array
    """
    lmbda = (len(mu)+1)/2
    return ghypmv.var(lmbda, chi, psi, mu, sigma, gamma)

def fit(x, alpha_bar=1, symmetric=False, standardize=False, nit=2000, reltol=1e-8,
        abstol=1e-7, silent=False, fmu=None, fsigma=None, fgamma=None, return_loglike=False):
    """
    Estimate the parameters of the hyperbolic distribution. We use the
    (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    Parameters
    ----------
    x: np.ndarray
        (n, p) array of n p-variate observations
    alpha_bar
        initial value of alpha_bar, a positive real number, where
        alpha_bar = sqrt(chi * psi)
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
    psi: float
    mu: np.ndarray
    sigma: np.ndarray
    gamma: np.ndarray
    log_likelihoods: list (returned if return_loglike = True)
    """

    opt_pars = {"lmbda": False, "alpha_bar": True, "mu": fmu is None,
                "sigma": fsigma is None, "gamma": fgamma is None}
    lmbda = (x.shape[1]+1)/2
    fit = ghypmv.fitghypmv(
        x, lmbda=lmbda, alpha_bar=alpha_bar, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars)
    
    chi, psi = ghypmv._alphabar2chipsi(fit["alpha_bar"], fit["lmbda"])
    if return_loglike:
        return fit["lmbda"], chi, psi, fit["mu"], fit["sigma"], fit["gamma"], fit["ll"]
    return fit["lmbda"], chi, psi, fit["mu"], fit["sigma"], fit["gamma"]


