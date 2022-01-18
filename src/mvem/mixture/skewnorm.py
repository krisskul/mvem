import numpy as np
from sklearn.cluster import KMeans
import scipy
from ..stats import multivariate_skewnorm as mvsn

def pdf(y, pi, mu, Sigma, lmbda):
    """pdf of multivariate skew-normal mixture model."""
    g = len(pi)
    density = 0
    for j in range(g):
        density = density + pi[j] * mvsn.pdf(y, mu[j].flatten(), Sigma[j], lmbda[j].flatten())
    return density

def rvs(pi, mu, Sigma, lmbda, size=1):
    """
    Random number generator of multivariate skew-normal
    mixture model.
    """

    w = np.random.choice(np.arange(len(pi)), p=pi, size=size)
    wi, wt = np.unique(w, return_counts=True)
    sample = []
    for j in range(len(wi)):
        i = wi[j]
        sample += [mvsn.rvs(mu[i], Sigma[i], lmbda[i], wt[j])]
    sample = np.concatenate(sample, axis=0)
    return sample

def predict(y, mu, Sigma, lmbda):
    """
    Predict probability of cluster belonging in multivariate
    skew-normal mixture model.
    """

    g = len(mu)
    prob = np.zeros((len(y), g))
    for i in range(g):
        prob[:, i] = mvsn.pdf(y, mu[i], Sigma[i], shape[i])
    prob = prob / np.sum(prob, axis=1)[:, None]
    return prob

def fit(y, g, mu=None, Sigma=None, shape=None, pi=None, kmeans_init=True,
        k_max_iter=50, k_n_init=1, max_iter=100, error=1e-4, uni_Gamma=False,
        criteria=True, group=True, obs_prob=True):
    """
    Fit a multivariate skew normal mixture model using an EM algorithm.
    
    Parameters
    ----------
    y: np.ndarray
        The response matrix of shape (n, p).
    g: int
        The number of cluster to be considered.
    mu: list
        (optional) A list of g arguments of vectors of initial values, with
        shape (p,), for the location parameters.
    Sigma: list
        (optional) A list of g arguments of matrices of initial values, with
        shape (p,p), for the scale parameters.
    shape: list
        (optional) A list of g arguments of vectors of initial values, with
        shape (p,), for the skewness parameters.
    pi: np.ndarray
        (optional) A vector of initial values, with length g, for the weights
        for each cluster. Must sum one.
    kmeans_init: bool
        If True, the initial values, which are not specified, are generated
        via k-means. _pi_ is computed via k-means, whether specified or not.
    k_max_iter: int
        Maximum number of iterations of the k-means algorithm for a single run.
        Default = 50
    k_n_init: int
        Number of time the k-means algorithm will be run with different centroid
        seeds. Default = 1.
    max_iter: int
        The maximum number of iterations of the EM algorithm. Default = 100.
    error: float
        The covergence maximum error for log-likelihood.
    uni_Gamma: bool
        If True, the Gamma parameters are restricted to be the same for all clusters.
    criteria: bool
        If True, log-likelihood, AIC, DIC, EDC and ICL will be calculated.
    group: bool
        If True, the vector with the classification of the response is returned.
    obs_prob: bool
        If TRUE, the posterior probability of each observation belonging to one
        of the g groups is reported.
        
    Returns
    -------
    dict
    """
    
    # Initialise parameters
    if g < 1: raise Exception("_g_ must be greater than 0.\n")
    n, p = y.shape
    
    if not kmeans_init:
        # Assure that parameters are correctly specified
        if len(mu) != len(Sigma) or len(mu) != len(pi) or len(mu) != len(shape):
            raise Exception("The size of the initial values are not compatible.\n")
        if np.sum(pi) != 1:
            raise Exception("Probability of _pi_ does not sum to 1.\n")
    else:
        # Initialise parameters if not specified
        if mu is not None: key_mu = False
        else: key_mu = True; mu = [0]*g
        if Sigma is not None: key_sig = False
        else: key_sig = True; Sigma = [0]*g
        if shape is not None: key_shp = False
        else: key_shp = True; shape = [0]*g

        # Assure correct parameter shape
        if (not key_mu and len(mu)!=g) or (not key_sig and Sigma.shape[0]!=g and Sigma.shape[1]!=g) or \
            (not key_shp and len(shape)!=g):
            raise Exception("The size of the initial values are not compatible.\n")

        if g > 1:
            # Use k-means to estimate parameters
            if key_mu:
                init = KMeans(n_clusters=g, max_iter=k_max_iter, n_init=k_n_init).fit(y)
            else:
                init = KMeans(n_clusters=g, init=np.r_[mu], max_iter=k_max_iter, n_init=k_n_init).fit(y)

            pi = np.unique(init.labels_, return_counts=True)[1] / (n)

            for j in range(g):
                if key_mu: mu[j] = init.cluster_centers_[[j]]
                if key_shp: shape[j] = np.sign(np.sum((y[init.labels_==j] - mu[j])**3, axis=0))
                if key_sig: Sigma[j] = np.cov(y[init.labels_==j].T)
        else:
            # Only one cluster ... use mean, cov and third moment to estimate parameters
            pi = 1
            if key_mu: mu = np.mean(y, axis=0)
            if key_sig: Sigma = np.cov(y.T)
            if key_shp: shape = np.sign(np.sum((y-mu)**3, axis=0))
                
    # Initialise parameters in (delta, gamma) space
    delta, Delta, Gamma = [0]*g, [0]*g, [0]*g
    for k in range(g):
        delta[k] = shape[k] / np.sqrt(1 + shape[k] @ shape[k])
        Delta[k] = scipy.linalg.sqrtm(Sigma[k]) @ delta[k]
        Gamma[k] = Sigma[k] - np.outer(Delta[k], Delta[k])

    # If uni_Gamma, force the Gamma parameters to be the same for all clusters
    if uni_Gamma:
        Gamma_uni = sum(Gamma) / g
        if g > 1:
            for k in range(g):
                Gamma[k] = Gamma_uni

    # Compute log-likelihood
    l = np.sum(np.log(pdf(y, pi, mu, Sigma, shape)))

    # Save starting values
    mu_old = mu
    Delta_old = Delta
    Gamma_old = Gamma
    criterion = 1
    count = 0
    l_old = l
    
    def _assure_not_zero(vec, eps=1e-10):
        inds = np.where(vec == 0)[0]
        if len(inds) > 0:
            vec[inds] = eps
        return vec
    
    # Run EM algorithm
    while criterion > error and count <= max_iter:
        count = count + 1
        tal, S1, S2, S3 = np.zeros((n, g)), np.zeros((n, g)), np.zeros((n, g)), np.zeros((n, g))

        # For each cluster
        for j in range(g):
            
            ### E-step: calculate ui, tui, tui2 ###
            diff = y-mu[j]
            Mtij2 = 1/(1 + Delta[j] @ np.linalg.inv(Gamma[j]) @ Delta[j])
            Mtij = np.sqrt(Mtij2)
            mutij = np.sum((Mtij2 * Delta[j] @ np.linalg.inv(Gamma[j])) * diff, axis=1)
            A = mutij / Mtij

            prob = scipy.stats.norm.cdf(A)
            prob = _assure_not_zero(prob)

            E = scipy.stats.norm.pdf(A) / prob
            u = np.ones(n)

            d1 = mvsn.pdf(y, mu[j].flatten(), Sigma[j], shape[j])
            d1 = _assure_not_zero(d1)
            d2 = pdf(y, pi, mu, Sigma, shape)
            d2 = _assure_not_zero(d2)

            tal[:,j] = d1 * pi[j] / d2
            S1[:,j] = tal[:,j]*u
            S2[:,j] = tal[:,j]*(mutij*u + Mtij*E)
            S3[:,j] = tal[:,j]*(mutij**2*u + Mtij2 + Mtij*mutij*E)

            ### M-step: update mu, Delta, Gama, sigma2 ###
            pi[j] = (1/n)*np.sum(tal[:,j])
            mu[j] = np.sum(S1[:,j][:, None] * y - S2[:, j][:, None] * Delta_old[j], axis=0) / np.sum(S1[:,j])

            diff = y - mu[j]
            Delta[j] = np.sum(S2[:, j][:, None] * diff, axis=0) / np.sum(S3[:, j])

            """sum2 = np.zeros((p,p))
            for i in range(n):
                diff2 = np.outer(diff[i], diff[i])
                delta2 = np.outer(Delta[j], Delta[j])
                tmp = np.outer(Delta[j], diff[i])
                sum2 = sum2 + S1[i,j]*diff2 - S2[i,j]*tmp - S2[i,j]*tmp.T + S3[i,j]*delta2"""
            diffp = np.repeat(diff, p, axis=1)
            diffpF = diffp.reshape(n, p, p, order="F")
            diffpC = diffp.reshape(n, p, p, order="C")
            # ... sum_{i=1}^n S1[i,j] * np.outer(diff[i], diff[i])
            tmp0 = np.sum(S1[:, j].reshape(n, 1, 1) * diffpC * diffpF, axis=0)
            # ... sum_{i=1}^n S3[i,j] * np.outer(Delta[j], Delta[j])
            delta2 = np.outer(Delta[j], Delta[j]).reshape(1, p, p)
            tmp1 = np.sum(delta2 * S3[:,j].reshape(-n, 1, 1), axis=0)
            # ... sum_{i=1}^n - S2[i,j]*tmp - S2[i,j]*tmp.T
            delta_diff = np.repeat(Delta[j], p).reshape(-p, p) * diffpF
            delta_diffT = np.rot90(delta_diff, -1, axes=(1,2))[:, :, ::-1]
            tmp2 = np.sum(-S2[:,j][:, None, None] * (delta_diff + delta_diffT), axis=0)
            # ...
            sum2 = tmp0 + tmp1 + tmp2

            Gamma[j] = sum2 / np.sum(tal[:,j])

            if not uni_Gamma:
                Sigma[j] = Gamma[j] + np.outer(Delta[j], Delta[j])
                shape[j] = (np.linalg.inv(scipy.linalg.sqrtm(Sigma[j])) @ Delta[j]) / np.sqrt(
                    1 - Delta[j] @ np.linalg.inv(Sigma[j]) @ Delta[j])
        # end(for)

        if uni_Gamma:
            GS = 0
            for j in range(g):
                GS = GS + np.kron(tal[:,j], Gamma[j])
                
            Gamma_uni = np.sum(GS.reshape(p, p, n, order="F"), axis=2) / n
            for j in range(g):
                Gamma[j] = Gamma_uni
                Sigma[j] = Gamma[j] + np.outer(Delta[j], Delta[j])
                shape[j] = (np.linalg.inv(scipy.linalg.sqrtm(Sigma[j])) @ Delta[j]) /\
                    np.sqrt(1 - Delta[j] @ np.linalg.inv(Sigma[j]) @ Delta[j])

        pi[g-1] = 1 - (np.sum(pi) - pi[g-1])
        zero_pos = np.where(pi==0)[0]
        if len(zero_pos) != 0:
            pi[zero_pos] = 1e-10
            pi[np.argmax(pi)] = np.max(pi) - np.sum(pi[zero_pos])

        l = np.sum(np.log(pdf(y, pi, mu, Sigma, shape)))
        criterion = np.abs(l/l_old-1)
        l_old = l
        mu_old = mu
        Delta_old = Delta
        Gamma_old = Gamma
    # end(while)
    
    if criteria:

        cl = np.argmax(tal,axis=1)
        icl = 0
        for j in range(g):
            icl = icl + np.sum(np.log(mvsn.pdf(y[cl==j], mu[j].flatten(), Sigma[j], shape[j])))

        if uni_Gamma:
            d = g*2*p + len(np.triu_indices(len(Sigma[0]))[0]) + (g-1) #mu + shape + Sigma + pi    
        else:
            d = g*(2*p + len(np.triu_indices(len(Sigma[0]))[0])) + (g-1) #mu + shape + Sigma + pi

        aic = -2*l + 2*d
        bic = -2*l + np.log(n)*d
        edc = -2*l + 0.2*np.sqrt(n)*d
        icl = -2*icl + np.log(n)*d
        obj_out = {"mu": mu, "Sigma": Sigma, "shape": shape, "pi": pi,# "nu" = nu,
                   "logLik": l, "aic": aic, "bic": bic, "edc": edc, "icl": icl,
                   "iter": count, "n": n, "group": cl}
    else:
        obj_out = {"mu": mu, "Sigma": Sigma, "shape": shape, "pi": pi,# "nu" = nu,
                   "iter": count, "n": n, "group": np.argmax(tal, axis=1)}

    if not group:
        obj_out.pop("group", None)

    obj_out["uni_Gamma"] = uni_Gamma    

    if obs_prob:
        tal[:, -1] = 1 - np.sum(tal[:, :(g-1)], axis=1)
        tal[tal<0] = 0.
        obj_out["obs_prob"] = tal

    return obj_out