import numpy as np
import scipy.stats
from scipy.special import gammaln, kv, digamma


def _gig_to_scipy_params_(lmbda, chi, psi):
    """Transform from (chi, psi) parameterisations to (loc, scale)"""
    p = lmbda
    b = np.sqrt(chi*psi)
    loc = 0
    scale = np.sqrt(chi/psi)
    return p, b, 0, scale

def _check_gig_pars(lmbda, chi, psi):
    if (lmbda < 0) and (chi <= 0 or psi < 0):
        raise Exception("If lambda < 0: chi must be > 0 and psi must be >= 0! \n" + \
                        "lambda = " + str(lmbda) + "; chi = " + str(chi) + "; psi = " + str(psi) + "\n")
    if (lmbda == 0) and (chi <= 0 or psi <= 0):
        raise Exception("If lambda == 0: chi must be > 0 and psi must be > 0! \n" + \
                        "lambda = " + str(lmbda) + "; chi = " + str(chi) + "; psi = " + str(psi) + "\n")
    if (lmbda > 0) and (chi < 0 or psi <= 0):
        raise Exception("If lambda > 0: chi must be >= 0 and psi must be > 0! \n" + \
                        "lambda = " + str(lmbda) + "; chi = " + str(chi) + "; psi = " + str(psi) + "\n")

def logpdf(x, lmbda=1, chi=1, psi=1):
    _check_gig_pars(lmbda, chi, psi)

    if psi==0:
        # inverse gamma (students t case)
        logdensity = scipy.stats.invgamma.logpdf(x, a=-lmbda, scale=0.5*chi)
    elif chi==0:
        # gamma (VG case)
        logdensity = scipy.stats.gamma.logpdf(x, a=lmbda, scale=2./psi)
    else:
        p, b, loc, scale = _gig_to_scipy_params_(lmbda, chi, psi)
        logdensity = scipy.stats.geninvgauss.logpdf(x, p=p, b=b, loc=loc, scale=scale)
                    
    return logdensity

# dgig
def pdf(x, lmbda=1, chi=1, psi=1):
    logdensity = logpdf(x, lmbda, chi, psi)
    return np.exp(logdensity)

def logcdf(x, lmbda=1, chi=1, psi=1):
    _check_gig_pars(lmbda, chi, psi)

    if psi==0:
        # inverse gamma (students t case)
        return scipy.stats.invgamma.logcdf(x, a=-lmbda, scale=0.5*chi)
    elif chi==0:
        # gamma (VG case)
        return scipy.stats.gamma.logcdf(x, a=lmbda, scale=2./psi)
    else:
        p, b, loc, scale = _gig_to_scipy_params_(lmbda, chi, psi)
        return scipy.stats.geninvgauss.logcdf(x, p=p, b=b, loc=loc, scale=scale)

# pgig
def cdf(x, lmbda=1, chi=1, psi=1):
    logcdf = logcdf(x, lmbda, chi, psi)
    return np.exp(logcdf)

def rvs(lmbda=1, chi=1, psi=1, size=1):
    _check_gig_pars(lmbda, chi, psi)

    if psi==0:
        # inverse gamma (students t case)
        return scipy.stats.invgamma.rvs(a=-lmbda, scale=0.5*chi, size=size)
    elif chi==0:
        # gamma (VG case)
        return scipy.stats.gamma.rvs(a=lmbda, scale=2./psi, size=size)
    else:
        p, b, loc, scale = _gig_to_scipy_params_(lmbda, chi, psi)
        return scipy.stats.geninvgauss.rvs(p=p, b=b, loc=loc, scale=scale, size=size)

def _besselM3(lmbda=9/2, x=2, logvalue=False):
    if np.all(np.abs(lmbda)==0.5):
        ## Simplified expression in case lambda == 0.5
        if not logvalue:
            res = np.sqrt(np.pi/(2*x))*np.exp(-x)
        else:
            res = 0.5*np.log(np.pi/(2*x))-x
    else:
        if not logvalue:
            res = kv(lmbda, x)
        else:
            res = np.log(kv(lmbda, x))
    return res

def expect(lmbda, chi, psi, func = "x"):
    """
    Compute the expectation E[f(x)|lmbda, chi, psi],
    where f(x) is one of ["x", "logx", "1/x"].
    """
    
    if func not in ["x", "logx", "1/x"]:
        raise Exception("_func_ must be one of ['x', 'logx', '1/x'], but is: " + str(func))

    lmbda = np.asarray(lmbda)
    chi = np.asarray(chi)
    psi = np.asarray(psi)
    
    #for i in range(len(lmbda)):
    #    _check_gig_pars(lmbda[i], chi[i], psi[i])

    chi_eps = 1e-8


    if func == "x":
        if np.all(psi == 0):
            # inv Gamma (Student-t)
            beta = 0.5 * chi
            alpha = -lmbda
            alpha[alpha <= 1] = np.nan # expectation is not defined if alpha <= 1
            return beta/(alpha-1)
        
        elif np.all(chi == 0):
            # gamma (VG)
            beta = 0.5 * psi
            alpha = lmbda
            return alpha/beta
        
        else:
            # gig (ghyp, hyp, NIG)
            chi[np.abs(chi) < chi_eps] = np.sign(chi[np.abs(chi) < chi_eps]) * chi_eps
            chi[chi == 0] = chi_eps

            alpha_bar = np.sqrt(chi * psi)
            term1 = 0.5 * np.log(chi / psi)
            term2 = _besselM3(lmbda + 1, alpha_bar, logvalue = True)
            term3 = _besselM3(lmbda, alpha_bar, logvalue = True)
        
            return np.exp(term1 + term2 - term3)
    
    elif func == "logx":
        if np.all(psi == 0):
            ## Inv Gamma -> Student-t
            beta = 0.5 * chi
            alpha = -lmbda
            return np.log(beta) - digamma(alpha)
        elif np.all(chi == 0):
            ## Gamma -> VG
            beta = 0.5 * psi
            alpha = lmbda
            return digamma(alpha) - np.log(beta)
        else:
            ## GIG -> ghyp, hyp, NIG
            chi[np.abs(chi) < chi_eps] = np.sign(chi[np.abs(chi) < chi_eps]) * chi_eps
            chi[chi == 0] = chi_eps

            # requires numerical calculations of the derivative of E[X**a] for a=sqrt(chi*psi)
            alpha_bar = np.sqrt(chi * psi)
            besselKnu = lambda x, alpha_bar: kv(x, alpha_bar)
            #Kderiv = np.concatenate([scipy.optimize.approx_fprime(lmbda, besselKnu, 1e-8, a) for a in alpha_bar])
            step = 1e-8
            Kderiv = (besselKnu(lmbda+step, alpha_bar) - besselKnu(lmbda-step, alpha_bar)) / (2*step)
            return 0.5 * np.log(chi/psi) + Kderiv / kv(lmbda, alpha_bar)
            
    elif func == "1/x":
        if np.all(psi == 0):
            # Inv Gamma -> Student-t
            beta = 0.5 * chi
            alpha = -lmbda
            return alpha / beta
        elif np.all(chi == 0):
            ## Gamma -> VG            
            print("Case 'chi == 0' and 'func = 1/x' is not implemented, using scipy function...")
            return scipy.stats.gamma.expect(lambda x: 1/x, args=(lmbda,), scale=2./psi)
        else:
            # GIG -> ghyp, hyp, NIG
            chi[np.abs(chi) < chi_eps] = np.sign(chi[np.abs(chi) < chi_eps]) * chi_eps
            chi[chi==0] = chi_eps

            alpha_bar = np.sqrt(chi * psi)
            term1 = -0.5 * np.log(chi / psi)
            term2 = _besselM3(lmbda - 1, alpha_bar, logvalue = True)
            term3 = _besselM3(lmbda, alpha_bar, logvalue = True)
            return np.exp(term1 + term2 - term3)
    

def var(lmbda, chi, psi):
    """
    Compute the variance of a GIG distribution with parameters
    (lmbda, chi, psi). When psi==0, the distribution is the inverse
    Gamma distribution, and when chi==0, the distribution is the Gamma
    distribution.
    
    Handles a vector of parameters, if the different distributions all
    belong to the same group. That is, if chi==0 for one set of parameters,
    it must be so for all sets of parameters. 
    """
    
    lmbda = np.asarray(lmbda)
    chi = np.asarray(chi)
    psi = np.asarray(psi)
    
    for i in range(len(lmbda)):
        _check_gig_pars(lmbda[i], chi[i], psi[i])

    chi_eps = 1e-8
        
    if np.all(psi == 0):
        ## Inv Gamma -> Student-t
        beta = 0.5 * chi
        alpha = -lmbda

        tmp_var = beta**2 / ((alpha - 1)**2 * (alpha - 2))
        tmp_var[alpha <= 2] = np.inf # variance is Inf if alpha <= 2
        return tmp_var

    elif np.all(chi == 0):
        ## Gamma -> VG
        beta = 0.5 * psi
        alpha = lmbda
        return alpha/(beta**2)
    else:
        ## GIG -> ghyp, hyp, NIG
        chi[np.abs(chi) < chi_eps] = np.sign(chi[np.abs(chi) < chi_eps]) * chi_eps
        chi[chi==0] = chi_eps

        alpha_bar = np.sqrt(chi * psi)
        term1 = 0.5 * np.log(chi / psi)
        term2 = _besselM3(lmbda + 1, alpha_bar, logvalue = True)
        term3 = _besselM3(lmbda, alpha_bar, logvalue = True)
        var_term1 = np.log(chi / psi)
        var_term2 = _besselM3(lmbda + 2, alpha_bar, logvalue = True)
        var_term3 = _besselM3(lmbda, alpha_bar, logvalue = True)
        
        return np.exp(var_term1 + var_term2 - var_term3) - np.exp(term1 + term2 - term3)**2