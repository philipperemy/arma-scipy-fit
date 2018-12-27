import numpy as np
from scipy.optimize import minimize


def _mse(p, t):
    score = np.mean(np.square(np.clip(p, -1e10, 1e10) - t))
    # score = np.mean((np.sign(p) * np.sign(t) + 1) / 2)
    return score


def fit(y: np.array,
        order: list,  # [1, 1] => ARMA(1,1)
        solver: str = 'Nelder-Mead',
        score_function=_mse,
        fit_x0=False):
    assert len(order) == 2
    n_time_series, nobs = y.shape
    num_steps = 0
    scores = []

    assert solver in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']

    if fit_x0:
        from statsmodels.tsa.arima_model import ARMA
        print('Estimating the parameters (statsmodels.tsa)...')
        k_ar = np.zeros(shape=(n_time_series, order[0],))
        k_ma = np.zeros(shape=(n_time_series, order[1],))
        for ii in range(y.shape[0]):
            model = ARMA(y[ii], order).fit(trend='nc', disp=0)
            k_ar[ii] = model.params[:order[0]]
            k_ma[ii] = model.params[order[0]:]
    else:
        k_ar = np.random.uniform(low=-1, high=1, size=(n_time_series, order[0],)) * 0.1
        k_ma = np.random.uniform(low=-1, high=1, size=(n_time_series, order[1],)) * 0.1
    parameters = np.stack([k_ar, k_ma])  # (2, num_time_series, order).
    parameters_shape = parameters.shape

    def predict_step(x, k_ar_0, k_ma_0):
        assert len(x.shape) == len(k_ar_0.shape) == len(k_ma_0.shape)
        assert x.shape[0] == k_ar_0.shape[0] == k_ma_0.shape[0]
        order_ar = order[0]
        order_ma = order[1]
        num_time_series = x.shape[0]
        noises = np.zeros_like(x)
        predictions = np.zeros_like(x)
        noises[:, 0:order_ma] = np.random.normal(size=(num_time_series, order_ma), scale=0.1)
        for t in range(order_ar, nobs):
            ar_term = np.sum(k_ar_0 * np.flip(x[:, t - order_ar:t], axis=1), axis=1)
            ma_term = np.sum(k_ma_0 * np.flip(noises[:, t - order_ma:t], axis=1), axis=1)

            predictions[:, t] = ar_term + ma_term
            noises[:, t] = x[:, t] - predictions[:, t]

        return predictions

    def optimization_step(coefficients):
        nonlocal num_steps
        nonlocal scores

        k_ar_0, k_ma_0 = np.reshape(coefficients, parameters_shape)
        predictions = predict_step(y, k_ar_0, k_ma_0)
        score = score_function(predictions, y)
        scores.append(score)

        print('AR')
        print(np.matrix(k_ar_0))
        print('MA')
        print(np.matrix(k_ma_0))
        print(str(num_steps).zfill(6), score)
        print('#' * 80)

        num_steps += 1
        return score

    np.set_printoptions(linewidth=150, precision=4, suppress=True)
    try:
        res = minimize(fun=optimization_step,
                       x0=parameters.flatten(),
                       method=solver,
                       options={'maxiter': 10000, 'disp': True})
    finally:
        np.set_printoptions(linewidth=150, precision=None, suppress=True)
    return res, scores
