import numpy as np
import scipy.stats
from scipy.special import gammaln, kv, digamma
from .gig import _besselM3
from . import gig
import warnings

def logpdf(x, lmbda, chi, psi, mu, sigma, gamma):

    ## Density of a multivariate generalized hyperbolic distribution.
    ## Covers all special cases as well.

    n, d = x.shape
    diff = x - mu
    det_sigma = np.linalg.det(sigma)
    inv_sigma = np.linalg.inv(sigma)

    Q = np.sum((diff @ inv_sigma) * diff, axis=1) # mahalanobis

    if np.sum(np.abs(gamma)) == 0:
        symm = True
        skewness_scaled = 0
        skewness_norm = 0
    else:
        symm = False
        skewness_scaled = (diff @ inv_sigma) @ gamma
        skewness_norm = (gamma @ inv_sigma) @ gamma

    out = np.nan

    if psi==0:
        lmbda_min_d_2 = lmbda - d/2.
        if symm:
            # symmetric student's t
            interm = chi + Q

            log_const_top = -lmbda * np.log(chi) + gammaln(-lmbda_min_d_2)
            log_const_bottom = d/2. * np.log(np.pi) + 0.5 * np.log(det_sigma) + gammaln(-lmbda)
            log_top = lmbda_min_d_2 * np.log(interm)

            out = log_const_top + log_top - log_const_bottom

        else:
            # symmetric student's t
            interm = np.sqrt((chi + Q) * skewness_norm)

            log_const_top = -lmbda * np.log(chi) - lmbda_min_d_2 * np.log(skewness_norm)
            log_const_bottom = d/2 * np.log(2 * np.pi) + 0.5 * np.log(det_sigma) + \
                gammaln(-lmbda) - (lmbda + 1) * np.log(2)
            log_top = _besselM3(lmbda_min_d_2, interm, logvalue = True) + skewness_scaled
            log_bottom = -lmbda_min_d_2 * np.log(interm)

            out = log_const_bottom + log_top - log_const_bottom - log_bottom

    elif psi > 0:
        lmbda_min_d_2 = lmbda - d/2.

        if chi > 0: # ghyp, hyp and NIG (symmetric and asymmetric)
            log_top = _besselM3(lmbda_min_d_2, np.sqrt((psi + skewness_norm) * (chi + Q)),
                                logvalue = True) + skewness_scaled
            log_bottom = -lmbda_min_d_2 * np.log(np.sqrt((psi + skewness_norm) * (chi + Q)))
            log_const_top = -lmbda/2 * np.log(psi * chi) + (d/2) * np.log(psi) - \
                lmbda_min_d_2 * np.log(1 + skewness_norm/psi)
            log_const_bottom = (d/2) * np.log(2*np.pi) + _besselM3(
                lmbda, np.sqrt(chi*psi), logvalue = True) + 0.5 * np.log(det_sigma)

            out = log_const_top + log_top - log_const_bottom - log_bottom

        elif chi == 0: # Variance gamma (symmetric and asymmetric)
            eps = 1e-8

            # Standardized observations that are close to 0 are set to 'eps'.
            if np.any(Q < eps):
                # If lambda == 0.5 * dimension, there is another singularity.
                if np.abs(lmbda_min_d_2) < eps:
                    raise Exception("Unhandled singularity: Some standardized observations are close to 0 (< " + \
                                    str(eps) + ") and lambda is close to 0.5 * dimension!\n")
                else:
                    Q[Q < eps] = eps
                    Q[Q == 0] = eps
                    warnings.warn("Singularity: Some standardized observations are close to 0 (< " + \
                                  str(eps) + ")!\nObservations set to " + str(eps) + ".\n")

            log_top = _besselM3(lmbda_min_d_2, np.sqrt((psi + skewness_norm) * (chi + Q)), logvalue=True) + skewness_scaled
            log_bottom = -lmbda_min_d_2 * np.log(np.sqrt((psi + skewness_norm) * (chi + Q)))
            log_const_top = d * np.log(psi)/2 + (1-lmbda) * np.log(2) - \
                lmbda_min_d_2 * np.log(1 + skewness_norm/psi)
            log_const_bottom = (d/2) * np.log(2 * np.pi) + gammaln(lmbda) + 0.5 * np.log(det_sigma)
            out = log_const_top + log_top - log_const_bottom - log_bottom
        else:
            out = np.nan

    return out

def pdf(x, lmbda, chi, psi, mu, sigma, gamma):
    return np.exp(logpdf(x, lmbda, chi, psi, mu, sigma, gamma))

def rvs(lmbda, chi, psi, mu, sigma, gamma, size):
    d = len(mu)
    Z = scipy.stats.norm.rvs(size=size*d).reshape((-size,d))
    A = np.linalg.cholesky(sigma)
    #if is_gaussian: return Z @ A + mu
    # else:
    W = gig.rvs(lmbda, chi, psi, size)
    return np.sqrt(W)[:, None] * (Z @ A) + mu + np.outer(W, gamma)