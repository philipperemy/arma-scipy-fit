import json
import numpy as np
import os
from argparse import ArgumentParser
from uuid import uuid4

from arma_scipy import fit


def get_script_arguments():
    args = ArgumentParser()
    args.add_argument('-s', '--solver', type=str, default='Nelder-Mead')
    args.add_argument('-x0', '--fit_x0', action='store_true')
    return args.parse_args()


def mse(p, t):
    score = np.mean(np.square(np.clip(p, -1e10, 1e10) - t))
    # score = np.mean((np.sign(p) * np.sign(t) + 1) / 2)
    return score


def main():
    """
    Main training function.
    """
    params = np.load('y.npz')
    y = params['y']
    order = params['order']
    est_params = params['est']
    true_ar = params['true_ar']
    true_ma = params['true_ma']

    args = get_script_arguments()
    solver = args.solver
    fit_x0 = args.fit_x0

    res, scores = fit(y, order, solver, mse, fit_x0)

    print('Estimation of the coefficients with the scipy package:')
    print(res.x)

    print('Estimation of the coefficients with the statsmodels.tsa (least squares) package:')
    print(est_params)

    print('True AR coefficients:')
    print(true_ar)

    print('True MA coefficients:')
    print(true_ma)

    results = {
        'solver': str(solver),
        'scores': list(scores),
        'x': list(res.x)
    }

    output_dir = 'out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, str(uuid4()) + '.json')
    with open(output_filename, 'w') as w:
        json.dump(obj=results, fp=w, indent=4, sort_keys=True)
    print('Results dumped in', output_filename)


if __name__ == '__main__':
    main()
