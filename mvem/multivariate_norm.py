from scipy.stats import multivariate_normal
import numpy as np

def fit(X):
    mean = np.mean(X, axis = 0)
    cov = np.cov(X.T)
    return mean, cov

def pdf(x, mean, cov, allow_singular=False):
    return multivariate_normal.pdf(x, mean, cov, allow_singular)

def logpdf(x, mean, cov, allow_singular=False):
    return multivariate_normal.logpdf(x, mean, cov, allow_singular)

def loglike(x, mean, cov):
    return np.sum(logpdf(x, mean, cov))

def cdf(x, mean, cov, allow_singular=False, maxpts=1000000, abseps=1e-5, releps=1e-5):
    maxpts = maxpts * x.shape[1]
    return multivariate_normal.cdf(x, mean, cov, allow_singular, maxpts, abseps, releps)

def logcdf(x, mean, cov, allow_singular=False, maxpts=1000000, abseps=1e-5, releps=1e-5):
    maxpts = maxpts * x.shape[1]
    return multivariate_normal.logcdf(x, mean, cov, allow_singular, maxpts, abseps, releps)

def rvs(mean, cov, size=1, random_state=None):
    return multivariate_normal.rvs(mean, cov, size, random_state)

def mean(mean, cov):
    return mean

def var(mean, cov):
    return cov