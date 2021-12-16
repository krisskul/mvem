import numpy as np
from . import multivariate_genhyperbolic as ghypmv

def logpdf(x, chi, psi, mu, sigma, gamma):
    return ghypmv.logpdf(x, -0.5, chi, psi, mu, sigma, gamma)

def pdf(x, chi, psi, mu, sigma, gamma):
    return ghypmv.pdf(x, -0.5, chi, psi, mu, sigma, gamma)

def rvs(chi, psi, mu, sigma, gamma, size):
    return ghypmv.rvs(-0.5, chi, psi, mu, sigma, gamma, size)

def fit(x, alpha_bar=1, symmetric=False, standardize=False, nit=2000, reltol=1e-8,
        abstol=1e-7, silent=False, fmu=None, fsigma=None, fgamma=None):
    
    opt_pars = {"lmbda": False, "alpha_bar": True, "mu": fmu is None,
                "sigma": fsigma is None, "gamma": fgamma is None}
    
    fit = ghypmv.fitghypmv(
        x, lmbda=-0.5, alpha_bar=alpha_bar, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars)
    
    return fit