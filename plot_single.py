import json
import matplotlib.pyplot as plt
import numpy as np

input_filename = 'out/444bba9f-dc84-4c38-bced-53f9cb1e9bac.json'

with open(input_filename, 'r') as r:
    output_json = json.load(r)

time_series_array = np.array(output_json['scores'])

plt.title('score = 1 - np.mean(((p > 0) & (t > 0)) | ((p < 0) & (t < 0)))')
plt.xlabel('Optimization Steps')
plt.ylabel('Score (lower is better)')

plt.plot(time_series_array)

plt.grid()
plt.legend(['Score'])
plt.show()
