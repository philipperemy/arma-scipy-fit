from setuptools import setup

setup(
    name='arma-scipy',
    version='1.1',
    description='Estimating coefficients of ARMA models with the Scipy package.',
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['arma_scipy'],
    install_requires=['scipy', 'numpy', 'statsmodels']
)
