import numpy as np


def mae(u, v):
    return np.mean(np.abs(u - v))


def main():
    """
    Main training function.
    """
    params = np.load('y.npz')
    y = params['y']
    order = params['order']
    nobs = len(y)

    def predict_step(x, k_ar_0, k_ma_0):
        order_ar = order[0]
        order_ma = order[1]
        noises = [np.random.normal(size=1, scale=0.1)] * order_ma
        predictions = [np.zeros(shape=(1,))] * order_ar
        for t in range(order_ar, nobs):
            pred = np.dot(k_ar_0, np.flip(x[t - order_ar:t])) + np.dot(k_ma_0, np.flip(noises[t - order_ma:t]))
            noise = x[t] - pred
            noises.append(noise)
            predictions.append(pred)
        predictions = np.transpose(predictions)
        return predictions

    num_steps = 0

    def score_function(p, t):
        score = np.mean(np.square(np.clip(p, -1e10, 1e10) - t))
        return score

    def optimization_step(coefficients):
        nonlocal num_steps
        nonlocal order
        k_ar_0 = coefficients[:order[0]]
        k_ma_0 = coefficients[order[0]:]
        predictions = predict_step(y, k_ar_0, k_ma_0)
        score = score_function(predictions, y)

        print(coefficients, score)

        num_steps += 1
        return score

    from scipy.optimize import minimize

    np.set_printoptions(linewidth=150, precision=2, suppress=True)

    solver = 'Nelder-Mead'  # Powell
    np.random.seed(123)
    k_ar = np.random.uniform(size=(order[0],)) * 0.1
    k_ma = np.random.uniform(size=(order[1],)) * 0.1
    res = minimize(optimization_step, np.concatenate([k_ar, k_ma]),
                   method=solver, options={'maxiter': 10000, 'disp': True})
    print(res.x)


if __name__ == '__main__':
    main()
