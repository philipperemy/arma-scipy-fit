import numpy as np
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess

np.random.seed(12346)
arparams = np.array([0.25, -0.5, 0.35, -0.15])  # np.array([0.25, -0.50])
maparams = np.array([0.50, -0.4, 0.78, 0.32])
ar = np.r_[1, -arparams]  # add zero-lag and negate
ma = np.r_[1, maparams]  # add zero-lag
arma_process = ArmaProcess(ar, ma)
y = arma_process.generate_sample(20000)
np.savez(file='y.npz', y=y, order=[len(arparams), len(maparams)])
model = ARMA(y, (len(arparams), len(maparams))).fit(trend='nc', disp=0)

print(model.params)

# import matplotlib.pyplot as plt
#
# plt.plot(y)
# plt.show()

# https://en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model
# Estimating coefficients >
# ARMA models in general can be, after choosing p and q, fitted by least
# squares regression to find the values of the parameters which minimize
# the error term. It is generally considered good practice to find the
# smallest values of p and q which provide an acceptable fit to the data.
# For a pure AR model the Yule-Walker equations may be used to provide a
# fit.

# https://en.wikipedia.org/wiki/Least_squares
# "Least squares" means that the overall solution minimizes the sum of the
# squares of the residuals made in the results of every single equation.
