from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='mvem',
    version='0.1.2',
    license='wtfpl',
    author='Kristoffer Skuland',
    author_email='kristoffer.skuland@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/krisskul/multivariate_em',
    keywords='maximum likelihood parameter estimation multivariate probability distribution EM algorithm',
    install_requires=[
          'scikit-learn',
          'numpy',
          'scipy>=1.6'
      ],
    description='Maximum likelihood estimation in multivariate probability distributions using EM algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
