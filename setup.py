from setuptools import setup, find_packages


setup(
    name='mvem',
    version='0.1',
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

)
