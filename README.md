# Maximum Likelihood Estimation in Multivariate Probability Distributions Using EM Algorithms

**mvem** is a Python package that provides maximum likelihood estimation methods for multivariate probability distributions using expectation–maximization (EM) algorithms. Additionally, it includes some functionality for fitting non-Gaussian multivariate mixture models. For fitting a wide range of univariate probability distributions, we refer to `scipy.stats`.

Currently included:
- normal (`mvem.stats.multivariate_norm`)
- skew normal (`mvem.stats.multivariate_skewnorm`)
- Student's *t* (`mvem.stats.multivariate_t`)
- normal-inverse Gaussian (`mvem.stats.multivariate_norminvgauss`)
- generalised hyperbolic skew Student's *t* (`mvem.stats.multivariate_genskewt`)
- generalised hyperbolic (`mvem.stats.multivariate_genhyperbolic`)
- hyperbolic (`mvem.stats.multivariate_hyperbolic`)
- variance-gamma (`mvem.stats.multivariate_vargamma`)

Multivariate mixture models currently included:
- skew normal (`mvem.mixture.skewnorm`)

## Where to get it
The source code is currently hosted on GitHub at: https://github.com/krisskul/mvem

Binary installers for the latest released version are available at the [Python Package Index (PyPI)](https://pypi.org/project/mvem).

```sh
pip install mvem
```

## Quickstart

```
from mvem.stats import multivariate_norminvgauss
# assume p-variate data x
params = multivariate_norminvgauss.fit(x)
```

## Requirements

```
numpy
scikit-learn
scipy>=1.6
```