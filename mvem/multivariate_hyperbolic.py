import numpy as np
from . import multivariate_genhyperbolic as ghypmv

def logpdf(x, chi, psi, mu, sigma, gamma):
    lmbda = (len(mu)+1)/2
    return ghypmv.logpdf(x, lmbda, chi, psi, mu, sigma, gamma)

def pdf(x, chi, psi, mu, sigma, gamma):
    lmbda = (len(mu)+1)/2
    return ghypmv.pdf(x, lmbda, chi, psi, mu, sigma, gamma)

def loglike(x, chi, psi, mu, sigma, gamma):
    return np.sum(logpdf(x, chi, psi, mu, sigma, gamma))

def rvs(chi, psi, mu, sigma, gamma, size):
    lmbda = (len(mu)+1)/2
    return ghypmv.rvs(lmbda, chi, psi, mu, sigma, gamma, size)

def mean(chi, psi, mu, sigma, gamma):
    lmbda = (len(mu)+1)/2
    return ghypmv.mean(lmbda, chi, psi, mu, sigma, gamma)

def var(chi, psi, mu, sigma, gamma):
    lmbda = (len(mu)+1)/2
    return ghypmv.var(lmbda, chi, psi, mu, sigma, gamma)

def fit(x, alpha_bar=1, symmetric=False, standardize=False, nit=2000, reltol=1e-8,
        abstol=1e-7, silent=False, fmu=None, fsigma=None, fgamma=None):
    
    opt_pars = {"lmbda": False, "alpha_bar": True, "mu": fmu is None,
                "sigma": fsigma is None, "gamma": fgamma is None}
    lmbda = (x.shape[1]+1)/2
    fit = ghypmv.fitghypmv(
        x, lmbda=lmbda, alpha_bar=alpha_bar, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars)
    
    return fit