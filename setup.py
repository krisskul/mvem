from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='mvem',
    version='0.1.1',
    license='wtfpl',
    author="Kristoffer Skuland",
    author_email='kristoffer.skuland@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/krisskul/multivariate_em',
    keywords='Maximum likelihood parameter estimation in multivariate distributions using EM algorithms',
    install_requires=[
          'scikit-learn',
          'numpy',
          'scipy'
      ],
    long_description=long_description,
    long_description_content_type='text/markdown',
)
