import json
import numpy as np
import pandas as pd
from glob import glob

all_scores = []

for output_filename in glob('out/**.json'):
    with open(output_filename, 'r') as r:
        output_json = json.load(r)
        scores = output_json['scores']
        all_scores.append(scores)
        print(scores[-1])

# max_len = max([len(s) for s in all_scores])
max_len = 1000

time_series_array = np.zeros(shape=(len(all_scores), max_len)) * np.nan

for i in range(len(all_scores)):
    time_series_array[i, 0:len(all_scores[i])] = all_scores[i][:max_len]

# https://stackoverflow.com/questions/50161140/how-to-plot-a-time-series-array-with-confidence-intervals-displayed-in-python
time_series_array[time_series_array > 10] = np.nanmedian(time_series_array)
time_series_df = pd.DataFrame(np.transpose(time_series_array))

std = time_series_df.std(axis=1)
mean = time_series_df.mean(axis=1)
best = time_series_df.min(axis=1)

# time_series_df.plot(legend=False)
import matplotlib.pyplot as plt

plt.title('Scipy vs Statsmodels')
plt.plot(mean, linewidth=2)
plt.plot(best, linewidth=1.5)
plt.fill_between(range(len(mean)), mean - 1 * std, mean + 1 * std, color='b', alpha=.2)
# plt.fill_between(range(len(mean)), mean - 2 * std, mean + 2 * std, color='b', alpha=.1)
plt.axhline(y=1.418, color='r', alpha=.3)
plt.grid()
plt.legend(['mean run (scipy.minimize)', 'best run (scipy.minimize)', 'statsmodels.tsa',
            ' +/- 1 std (scipy.minimize)'])  # , ' +/- 2 std'])
plt.show()
