import numpy as np
from scipy.optimize import minimize


def _mse(p, t):
    score = np.mean(np.square(np.clip(p, -1e10, 1e10) - t))
    # score = np.mean((np.sign(p) * np.sign(t) + 1) / 2)
    return score


def predict(x_, ar, ma,
            preprocessing_function=None,
            postprocessing_function=None):
    assert len(x_.shape) == len(ar.shape) == len(ma.shape)
    assert x_.shape[0] == ar.shape[0] == ma.shape[0]
    nobs = x_.shape[-1]
    if preprocessing_function is not None:
        x = np.array(preprocessing_function(x_))
    else:
        x = x_
    order_ar = ar.shape[-1]
    order_ma = ma.shape[-1]
    num_time_series = x.shape[0]
    noises = np.zeros_like(x)
    predictions = np.zeros_like(x)
    noises[:, 0:order_ma] = np.random.normal(size=(num_time_series, order_ma), scale=0.1)
    for t in range(order_ar, nobs):
        ar_term = np.sum(ar * np.flip(x[:, t - order_ar:t], axis=1), axis=1)
        ma_term = np.sum(ma * np.flip(noises[:, t - order_ma:t], axis=1), axis=1)

        predictions[:, t] = ar_term + ma_term
        noises[:, t] = x[:, t] - predictions[:, t]

    if postprocessing_function is not None:
        predictions = postprocessing_function(predictions)
    return predictions


def fit(y: np.array,
        order: list,  # [1, 1] => ARMA(1,1)
        solver: str = 'Nelder-Mead',
        score_function=_mse,
        fit_x0=False,
        preprocessing_function=None,
        postprocessing_function=None,
        ar_init_function=None,
        ma_init_function=None,
        verbose=True):
    assert len(order) == 2
    assert solver in ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP']
    n_time_series, nobs = y.shape
    num_steps = 0
    scores = []

    def default_init(shape):
        return np.random.uniform(low=-1, high=1, size=shape) * 0.1

    def ar_init(shape):
        return ar_init_function(shape) if ar_init_function is not None else default_init(shape)

    def ma_init(shape):
        return ma_init_function(shape) if ma_init_function is not None else default_init(shape)

    if fit_x0:
        from statsmodels.tsa.arima_model import ARMA
        print('Estimating the parameters (statsmodels.tsa)...')
        k_ar = np.zeros(shape=(n_time_series, order[0],))
        k_ma = np.zeros(shape=(n_time_series, order[1],))
        y_ = np.array(preprocessing_function(y))
        for ii in range(y_.shape[0]):
            try:
                model = ARMA(y_[ii], order).fit(trend='nc', disp=0)
                k_ar[ii] = model.params[:order[0]]
                k_ma[ii] = model.params[order[0]:]
            except Exception:
                k_ar[ii] = ar_init((order[0],))
                k_ma[ii] = ma_init((order[1],))
    else:
        k_ar = ar_init((n_time_series, order[0],))
        k_ma = ma_init((n_time_series, order[1],))
    parameters = np.stack([k_ar, k_ma])  # (2, num_time_series, order).
    parameters_shape = parameters.shape

    def optimization_step(coefficients):
        nonlocal num_steps
        nonlocal scores

        k_ar_0, k_ma_0 = np.reshape(coefficients, parameters_shape)
        predictions = predict(y, k_ar_0, k_ma_0, preprocessing_function, postprocessing_function)
        score = score_function(predictions, y)
        scores.append(score)

        if verbose and num_steps % 100 == 0:
            print(str(num_steps).zfill(6), score)
            print('AR')
            print(np.matrix(k_ar_0).flatten())
            print('MA')
            print(np.matrix(k_ma_0).flatten())
            print('#' * 80)
        num_steps += 1
        return score

    np.set_printoptions(linewidth=150, precision=4, suppress=True)
    print('Estimating the parameters (scipy.minimize)...')
    try:
        res = minimize(fun=optimization_step,
                       x0=parameters.flatten(),
                       method=solver,
                       options={'maxiter': 10000, 'disp': True})
    finally:
        np.set_printoptions(linewidth=150, precision=None, suppress=True)
    return res, scores, np.reshape(res.x, parameters_shape)
