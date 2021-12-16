# Maximum Likelihood Estimation in Multivariate Probability Distributions Using EM Algorithms

To my surprise, ```scipy.stats``` include very few multivariate probability distributions, and neither of the implemented distributions has an associated parameter estimation method. In contrast, R (:heart_eyes:) has more or less everything a statistician desires. Hence, if you are stuck with Python, this repository attempts to include parameter estimation methods for the most common continuous probability distributions using EM (or modified EM) algorithms.

Currently included:
- normal (```mvem.multivariate_norm```)
- skew normal (```mvem.multivariate_skewnorm```)
- Student's *t* (```mvem.multivariate_t```)
- normal-inverse Gaussian (```mvem.multivariate_norminvgauss```)
- generalised hyperbolic skew Student's *t* (```mvem.multivariate_genskewt```)
- generalised hyperbolic (```mvem.multivariate_genhyperbolic```)
- hyperbolic (```mvem.multivariate_hyperbolic```)
- variance-gamma (```mvem.multivariate_norm```)

## Example

!["alt"](assets/msn.png "Title")

## Current library structure:

```
mvem
|_ multivariate_normal.py
   |_ fit, pdf, logpdf, rvs, mean, var
|_ multivariate_t.py
   |_ fit, pdf, logpdf, rvs, mean, var
...
```

## Requirements

```
numpy
scipy>=1.6
```