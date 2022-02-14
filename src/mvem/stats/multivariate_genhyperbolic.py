import numpy as np
import scipy.stats
from scipy.special import gammaln, kv, digamma
from mvem.stats.gig import _besselM3, _check_gig_pars
from mvem.stats import gig
import warnings

def logpdf(x, lmbda, chi, psi, mu, sigma, gamma):
    """
    Log-probability density function of the generalised hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
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
    """
    Probability density function of the generalised hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
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
    return np.exp(logpdf(x, lmbda, chi, psi, mu, sigma, gamma))

def loglike(x, lmbda, chi, psi, mu, sigma, gamma):
    """
    Log-likelihood function of the generalised hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
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
    return np.sum(logpdf(x, chi, lmbda, psi, mu, sigma, gamma))

def rvs(lmbda, chi, psi, mu, sigma, gamma, size=1):
    """
    Random number generator of the generalised hyperbolic distribution.
    We use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
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
    d = len(mu)
    Z = scipy.stats.norm.rvs(size=size*d).reshape((-size,d))
    A = np.linalg.cholesky(sigma)
    #if is_gaussian: return Z @ A + mu
    # else:
    W = gig.rvs(lmbda, chi, psi, size)
    return np.sqrt(W)[:, None] * (Z @ A) + mu + np.outer(W, gamma)

def mean(lmbda, chi, psi, mu, sigma, gamma):
    """
    Mean function of the generalised hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
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
    if psi==0: # gig -> invgamma        
        if lmbda > -1:
            raise Exception("Mean for psi==0 is only defined for lambda < -1")
        alpha = -lmbda
        beta = 0.5*chi
        EW = beta/(alpha-1)
    elif psi>0:
        if chi==0: # gig -> gamma
            alpha = lmbda 
            beta = 0.5*psi
            EW = beta/alpha
        elif chi>0: # gig
            alpha = np.sqrt(chi*psi)
            EW = np.sqrt(chi/psi) * kv(lmbda+1, alpha) / kv(lmbda, alpha)

    return mu + EW*gamma

def var(lmbda, chi, psi, mu, sigma, gamma):
    """
    Variance function of the generalised hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param lmbda: Univariate parameter.
    :type lmbda: float
    :param chi: Univariate parameter.
    :type chi: float
    :param psi: Univariate parameter.
    :type psi: float
    :param mu: Location parameter with shape (p,).
    :type mu: np.ndarray
    :param sigma: A positive semi-definite array with shape (p, p).
    :type sigma: np.ndarray
    :param gamma: Parameter with shape (p,).
    :type gamma: np.ndarray
    :return: The variance of the specified distribution.
    :rtype: np.ndarray with shape (p,).
    """
    if psi==0: # gig -> invgamma
        if lmbda > -2:
            raise Exception("Var for psi==0 is only defined for lambda < -2")
        alpha = -lmbda
        beta = 0.5*chi
        EW = beta/(alpha-1)
        VarW = beta**2 / ((alpha-1)**2 * (alpha-2))
    elif psi>0:
        if chi==0: # gig -> gamma
            alpha = lmbda 
            beta = 0.5*psi
            EW = beta/alpha
            VarW = alpha/(beta**2)
        elif chi>0: # gig
            alpha = np.sqrt(chi*psi)
            EW = np.sqrt(chi/psi) * kv(lmbda+1, alpha) / kv(lmbda, alpha)
            EW2 = (chi/psi) * kv(lmbda+2, alpha) / kv(lmbda, alpha)
            VarW = EW2 - EW**2

    return VarW * np.outer(gamma, gamma) + EW * sigma

def _alphabar2chipsi(alpha_bar, lmbda):
    """
    From (alpha_bar, lambda) parameterisation to (chi, psi)
    """
    psi = alpha_bar * kv(lmbda+1, alpha_bar) / kv(lmbda, alpha_bar)
    chi = alpha_bar**2 / psi
    return chi, psi

def t_optfunc(thepars, delta_sum, xi_sum, n_rows):
    """
    Log-likelihood function of the inverse gamma distribution
    """
    nu = -2 * (-1 - exp(thepars))
    term1 = -n.rows * nu * np.log(nu/2 - 1)/2
    term2 = (nu/2 + 1) * xi_sum + (nu/2 - 1) * delta_sum
    term3 = n_rows * gammaln(nu/2)
    out = term1 + term2 + term3
    return out

# loglikelihood function of the gamma distribution
def vg_optfunc(thepars, xi_sum, eta_sum, n_rows):
    """
    Log-likelihood function of the gamma distribution
    """

    thepars = np.exp(thepars)
    term1 = n_rows * (thepars * np.log(thepars) - gammaln(thepars))
    term2 = (thepars - 1) * xi_sum - thepars * eta_sum
    out = -(term1 + term2)
    return out

def gig_optfunc(thepars, mix_pars_fixed, pars_order, delta_sum, eta_sum, xi_sum, n_rows):
    """
    Log-likelihood function of the generalized inverse gaussian distribution
    """
    out = np.nan
    tmp_pars = np.concatenate([thepars, mix_pars_fixed])
    lmbda = tmp_pars[pars_order=="lmbda"]
    alpha_bar = np.exp(tmp_pars[pars_order=="alpha_bar"])    
    chi, psi = _alphabar2chipsi(alpha_bar, lmbda)
    if lmbda < 0 and psi == 0: # t
        out = t_optfunc(lmbda, delta_sum, xi_sum, n_rows)
    elif lmbda > 0 and chi == 0: # VG
        out = vg_optfunc(lmbda, xi_sum, eta_sum, n_rows)
    else: # ghyp, hyp, NIG
        term1 = (lmbda - 1) * xi_sum
        term2 = -chi * delta_sum/2
        term3 = -psi * eta_sum/2
        term4 = -n_rows * lmbda * np.log(chi)/2 + n_rows * lmbda * np.log(psi)/2 - \
            n_rows * _besselM3(lmbda, np.sqrt(chi * psi), logvalue = True)
        out = -(term1 + term2 + term3 + term4)
    return out

def _check_norm_pars(mu, sigma, gamma, dimension):
    """
    Check normal and skewness parameters  for consistency
    """
    if len(mu) != dimension:
        raise Exception("Parameter 'mu' must be of length " + str(dimension) + "!")
    if len(gamma) != dimension:
        raise Exception("Parameter 'gamma' must be of length " + str(dimension) + "!")
    if dimension > 1: # MULTIVARIATE
        if len(sigma.shape) != 2 or sigma.shape[0] != dimension or sigma.shape[1] != dimension:
            raise Exception("'sigma' must be a quadratic matrix with dimension " +\
                 str(dimension) + " x " + str(dimension) + "!")    
    else: # UNIVARIATE
        if len(sigma) != dimension:
            raise Exception("Parameter 'sigma' must be a scalar!")

def fitghypmv(
    x, lmbda = 1, alpha_bar = 1, mu = None, sigma = None, gamma = None, symmetric = False,
    standardize = False, nit = 2000, reltol = 1e-8, abstol = 1e-7, silent = False,
    opt_pars = {"lmbda": True, "alpha_bar": True, "mu": True, "sigma": True, "gamma": True}
):
    """
    Estimate the parameters of the generalised hyperbolic distribution. We use
    the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation. Covers all special
    cases as well.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: The intial value of lmbda. Defaults to 1.
    :tpe lmbda: float, optional
    :param alpha_bar: The initial value of alpha_bar. Defaults to 1.
    :type alpha_bar: float, optional
    :param mu: Optional initial value of mu, an array of shape (p,)
    :type mu: np.ndarray, optional
    :param sigma: Optional initial value of sigma, an array of shape (p,p)
    :type sigma: np.ndarray, optional
    :param gamma: Optional initial value of gamma, an array of shape (p,)
    :type gamma: np.ndarray, optional
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
    :param opt_pars: A dict of (lmbda, alpha_bar, mu, sigma, gamma) with boolean values,
        denoting if the parameters should be estimated or fixed to their initial value.
        Defaults to fitting all parameters.
    :type opt_pars: dict, optional
    :return: The fitted parameters (<float> lmbda, <float> alpha_bar, <array> mu, <array> sigma,
        <array> gamma) and a list of log-likelihood values at each iteration of the EM algorithm,
        the number of performed iterations, whether the algorithm coverged, and the final AIC.
    :rtype: dict
    """

    n, d = x.shape
    chi, psi = _alphabar2chipsi(alpha_bar, lmbda)

    # check parameters of the mixing distribution for consistency
    _check_gig_pars(lmbda, chi, psi)

    # .check.opt.pars(opt.pars, symmetric)
    #opt.pars <- .check.opt.pars(opt.pars, symmetric)

    m1 = np.mean(x, axis=0)
    center = x-m1

    if mu is None: mu = m1
    if (gamma is None) or symmetric: gamma = np.zeros(d)
    if symmetric: opt_pars["gamma"] = False
    if sigma is None: sigma = np.cov(x.T)

    # check normal and skewness parameters  for consistency
    _check_norm_pars(mu, sigma, gamma, d)

    if standardize:
        # data will be standardized and initial values will be adapted    
        tmp_mean = np.mean(x, axis=0)
        sigma_chol = np.linalg.cholesky(np.linalg.inv(np.cov(x.T))).T
        x = x - tmp_mean
        x = x @ sigma_chol
        sigma = sigma_chol.T @ sigma @ sigma_chol
        gamma = sigma_chol @ gamma
        mu = sigma_chol @ (mu - tmp_mean)

    # Initialize fitting loop
    i = 0
    rel_closeness = 100
    abs_closeness = 100
    tmp_fit = {"convergence": 0, "message": None}
    chi, psi = _alphabar2chipsi(alpha_bar, lmbda)
    ll = np.sum(logpdf(x, lmbda=lmbda, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))

    ll_save = []
    lmbda_save_save = []
    chi_save = []
    psi_save = []
    mu_save = []
    sigma_save = []
    gamma_save = []

    ## Start interations
    while (abs_closeness > abstol) and (rel_closeness > reltol) and (i < nit):
        i = i + 1

        ##<------------------------ E-Step: EM update ------------------------------>
        ## The parameters mu, sigma and gamma become updated

        inv_sigma = np.linalg.inv(sigma)

        diff = x-mu
        Q = np.sum((diff @ inv_sigma) * diff, axis=1)
        Offset = (gamma @ inv_sigma) @ gamma
        delta = gig.expect(lmbda-d/2, Q+chi, psi+Offset, func = "1/x")
        delta_bar = np.mean(delta)
        eta = gig.expect(lmbda-d/2, Q+chi, psi+Offset, func = "x")
        eta_bar = np.mean(eta)

        if opt_pars["gamma"]:
            gamma = -(delta @ center) / (n*delta_bar*eta_bar - n)

        if opt_pars["mu"]:
            mu = ((delta @ x)/n - gamma) / delta_bar

        diff = x - mu
        tmp = delta[:, None] * diff
        if opt_pars["sigma"]:
            sigma <- (tmp.T @ diff)/n - np.outer(gamma, gamma) * eta_bar

        ##<------------------------ M-Step: EM update ------------------------------>
        ## Maximise the conditional likelihood function and estimate lambda, chi, psi

        inv_sigma = np.linalg.inv(sigma)
        Q = np.sum((diff @ inv_sigma) * diff, axis=1)
        Offset = gamma @ inv_sigma @ gamma
        xi_sum = np.sum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="logx"))

        if alpha_bar==0 and lmbda > 0 and not opt_pars["alpha_bar"] and opt_pars["lmbda"]:
            ##<------  VG case  ------>
            eta_sum = np.sum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="x"))
            tmp_fit = scipy.optimize.minimize(vg_optfunc, np.log(lmbda), args=(eta_sum, xi_sum, n))
            lmbda = np.exp(tmp_fit["x"])

        elif alpha_bar==0 and lmbda<0 and not opt_pars["alpha_bar"] and opt_pars["lmbda"]:
            ##<------  Student-t case  ------>
            delta_sum = np.sum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="1/x"))
            tmp_fit = scipy.optimize.minimize(t_optfunc, np.log(-1 - lmbda), args=(delta_sum, xi_sum, n))
            lmbda = (-1 - exp(tmp_fit["x"]))

        elif opt_pars["lmbda"] or opt_pars["alpha_bar"]:
            ##<------  ghyp, hyp, NIG case  ------>
            delta_sum = np.sum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="1/x"))
            eta_sum = np.sum(gig.expect(lmbda-d/2, Q+chi, psi+Offset, func="x"))

            mix_pars = {"lmbda": lmbda, "alpha_bar": np.log(alpha_bar)}
            thepars = {x: mix_pars[x] for x in ["lmbda", "alpha_bar"] if opt_pars[x]}
            mix_pars_fixed = {x: mix_pars[x] for x in ["lmbda", "alpha_bar"] if not opt_pars[x]}
            pars_order = np.array(list(thepars.keys()) + list(mix_pars_fixed.keys()))
            thepars_v = np.array(list(thepars.values())).flatten()
            mix_pars_fixed_v = np.array(list(mix_pars_fixed.values())).flatten()

            tmp_fit = scipy.optimize.minimize(gig_optfunc, thepars_v, args=(
                mix_pars_fixed_v, pars_order, delta_sum, eta_sum, xi_sum, n))
            par = np.concatenate([tmp_fit["x"], mix_pars_fixed_v])

            lmbda = par[pars_order == "lmbda"]
            alpha_bar = np.exp(par[pars_order == "alpha_bar"])


        chi, psi = _alphabar2chipsi(alpha_bar, lmbda)

        ## Test for convergence
        ll_old = ll
        ll = np.sum(logpdf(x, lmbda=lmbda, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))

        abs_closeness = np.abs(ll - ll_old)
        rel_closeness = np.abs((ll - ll_old)/ll_old)

        if not np.isfinite(abs_closeness) or not np.isfinite(rel_closeness):
            warnings.warn("fit.ghypmv: Loglikelihood is not finite! Iteration stoped!\n" +\
                          "Loglikelihood :" + str(ll))
            break

        if not silent:
            print("iter: %i; rel.closeness %.6f; log-likelihood %.6f; alpha.bar %.6f; lambda %.6f" % (
                i, rel_closeness, ll, alpha_bar, lmbda))

        ## Assign current values of optimization parameters
        ## to backups of the nesting environment.
        # ...
        ll_save += [ll]
        lmbda_save_save += [lmbda]
        chi_save += [chi]
        psi_save += [psi]
        mu_save += [mu]
        sigma_save += [sigma]
        gamma_save += [gamma]

    ## END OF WHILE LOOP
    """
    conv <- tmp.fit$convergence
    conv.type <- tmp.fit$message
    if(is.null(tmp.fit$message)){
        conv.type <- ""
    }else{
        conv.type <- paste("Message from 'optim':", tmp.fit$message)
    }
    """

    converged = False
    if i<nit and np.isfinite(rel_closeness) and np.isfinite(abs_closeness): converged = True

    if(standardize):
        inv_sigma_chol = np.linalg.inv(sigma_chol)
        mu = inv_sigma_chol @ mu + tmp_mean    
        sigma = inv_sigma_chol.T @ sigma @ inv_sigma_chol
        gamma = inv_sigma_chol @ gamma

        chi, psi = _alphabar2chipsi(alpha_bar, lmbda)    
        ll = np.sum(logpdf(x, lmbda=lmbda, chi=chi, psi=psi, mu=mu, sigma=sigma, gamma=gamma))


    nbr_fitted_params = opt_pars["alpha_bar"] + opt_pars["lmbda"] + d * (opt_pars["mu"]+opt_pars["gamma"]) +\
        (d/2)*(d+1)*opt_pars["sigma"]

    aic = -2 * ll + 2 * nbr_fitted_params

    return {"lmbda": lmbda, "alpha_bar": alpha_bar, "mu": mu, "sigma": sigma, "gamma": gamma,
            "ll": ll_save, "n_iter": i, "converged": converged, "aic": aic}



def fit(x, lmbda=1, alpha_bar=1, symmetric=False, standardize=False, nit=2000,
        reltol=1e-8, abstol=1e-7, silent=False, flambda=None, falpha_bar=None,
        fmu=None, fsigma=None, fgamma=None, return_loglike=False):
    """
    Estimate the parameters of the generalised hyperbolic distribution. We
    use the (lmbda, chi, psi, mu, sigma, gamma)-parameterisation.

    :param x: An array of shape (n, p) containing n observations of some
        p-variate data with n > p.
    :type x: np.ndarray
    :param lmbda: The intial value of lmbda. Defaults to 1.
    :tpe lmbda: float, optional
    :param alpha_bar: The initial value of alpha_bar. Defaults to 1.
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
    :param flmbda: If flmbda!=None, force lmbda to flmbda. Defaults to None.
    :type flmbda: np.ndarray, optional
    :param falpha_bar: If falpha_bar!=None, force alpha_bar to falpha_bar. 
        Defaults to None.
    :type falpha_bar: np.ndarray, optional
    :param fmu: If fmu!=None, force mu to fmu. Defaults to None.
    :type fmu: np.ndarray, optional
    :param fsigma: If fsigma!=None, force sigma to fsigma. Defaults to None.
    :type fsigma: np.ndarray, optional
    :param fgamma: If fgamma!=None, force gamma to fgamma. Defaults to None.
    :type fgamma: np.ndarray, optional
    :param return_loglike: Return a list of log-likelihood values at each iteration. 
        Defaults to False.
    :type return_loglike: np.ndarray, optional
    :return: The fitted parameters (<float> lmbda, <float> chi, <float> psi, <array> mu,
        <array> sigma, <array> gamma). Also returns a list of log-likelihood values at
        each iteration of the EM algorithm if ``return_loglike=True``.
    :rtype: tuple
    """

    opt_pars = {"lmbda": flambda is None, "alpha_bar": falpha_bar is None,
                "mu": fmu is None, "sigma": fsigma is None, "gamma": fgamma is None}

    lmbda = lmbda if flambda is None else flambda
    alpha_bar = alpha_bar if falpha_bar is None else falpha_bar

    fit = fitghypmv(
        x, lmbda=lmbda, alpha_bar=alpha_bar, mu=fmu, sigma=fsigma, gamma=fgamma,
        symmetric=symmetric, standardize=standardize, nit=nit, reltol=reltol,
        abstol=abstol, silent=silent, opt_pars=opt_pars)

    if fit["alpha_bar"] != 0:
        chi, psi = _alphabar2chipsi(fit["alpha_bar"], fit["lmbda"])
    elif fit["alpha_bar"]==0 and fit["lmbda"] > 0:
         chi, psi = 0, 2 * fit["lmbda"]
    elif fit["alpha_bar"]==0 and fit["lmbda"] < 0:
         chi, psi = -2 * (fit["lmbda"] + 1), 0

    if return_loglike:
        return fit["lmbda"], chi, psi, fit["mu"], fit["sigma"], fit["gamma"], fit["ll"]
    return fit["lmbda"], chi, psi, fit["mu"], fit["sigma"], fit["gamma"]

