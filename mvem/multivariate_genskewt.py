import numpy as np
from . import multivariate_genhyperbolic as ghypmv

def logpdf(x, lmbda, chi, mu, sigma, gamma):
    return ghypmv.logpdf(x, lmbda, chi, 0, mu, sigma, gamma)

def pdf(x, lmbda, chi, mu, sigma, gamma):
    return ghypmv.pdf(x, lmbda, chi, 0, mu, sigma, gamma)

def rvs(lmbda, chi, mu, sigma, gamma, size):
    return ghypmv.rvs(lmbda, chi, 0, mu, sigma, gamma, size)

def mean(lmbda, chi, mu, sigma, gamma):
    return ghypmv.mean(lmbda, chi, 0, mu, sigma, gamma)

def var(lmbda, chi, mu, sigma, gamma):
    return ghypmv.var(lmbda, chi, 0, mu, sigma, gamma)

def fit(x, lmbda=-1, symmetric=False, standardize=False, nit=2000, reltol=1e-8,
        abstol=1e-7, silent=False, fmu=None, fsigma=None, fgamma=None):
    
    opt_pars = {"lmbda": True, "alpha_bar": False, "mu": fmu is None,
                "sigma": fsigma is None, "gamma": fgamma is None}
    
    fit = ghypmv.fitghypmv(
        x, lmbda=lmbda, alpha_bar=0, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars)
    
    return fit