# ARMA w. Scipy
Estimating coefficients of ARMA models with the Scipy package.


```
################################################################################
Optimization terminated successfully.
         Current function value: 1.432208
         Iterations: 508
         Function evaluations: 788
Estimation of the coefficients with the scipy package:
[ 0.2235 -0.5872  0.3143 -0.1805  0.167  -0.0464  0.6528  0.224 ]
Estimation of the coefficients with the statsmodels.tsa (least squares) package:
[ 0.237  -0.4998  0.3467 -0.128   0.1542 -0.1467  0.6244  0.2245]
True AR coefficients:
[ 0.25 -0.5   0.35 -0.15]
True MA coefficients:
[ 0.5  -0.4   0.78  0.32]
```
