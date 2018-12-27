# ARMA w. Scipy
Estimating coefficients of ARMA models with the Scipy package.


```
000847 [ 0.1499 -0.5917  0.28   -0.1156  0.2239 -0.0172  0.639   0.1996] 1.4415552801683131
Optimization terminated successfully.
         Current function value: 1.441535
         Iterations: 538
         Function evaluations: 848
Estimation of the coefficients with the scipy package:
[ 0.1499 -0.5917  0.28   -0.1156  0.2239 -0.0172  0.639   0.1996]
Estimation of the coefficients with the statsmodels.tsa (least squares) package:
[ 0.237  -0.4998  0.3467 -0.128   0.1542 -0.1467  0.6244  0.2245]
True AR coefficients:
[ 0.25 -0.5   0.35 -0.15]
True MA coefficients:
[ 0.5  -0.4   0.78  0.32]
```

<p align="center">
  <img src="misc/arma_44_fit.png" width="500">
</p>
