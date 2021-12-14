# Maximum Likelihood Estimation of Parameters in Multivariate Probability Distributions Using the EM Algorithm

For some reason, ```scipy.stats``` include very few multivariate probability distributions, and neither of the implemented distributions has an associated parameter estimation method. In contrast, R (:heart_eyes:) has more or less everything a statistician desires. Hence, if you are stuck with Python, this repository attempts to include parameter estimation methods for the most common continuous probability distributions using EM (or modified EM) algorithms.

Currently included:
- multivariate normal
- multivariate skew normal
- multivariate student's t

In time, I will (hopefully) add methods for, e.g., multivariate NIG, EMG, skew Student's t, etc. 

## Current library structure:

```
mvem
|_ multivariate_normal.py
   |_ fit, pdf, logpdf, rvs, mean, var, loglike, ...
|_ multivariate_t.py
   |_ fit, pdf, logpdf, rvs, mean, var, loglike, ...
|_ multivariate_skewnormal.py
   |_ fit, pdf, logpdf, rvs, mean, var, loglike, ...
```

## Requirements

```
numpy
scipy>=1.6
```