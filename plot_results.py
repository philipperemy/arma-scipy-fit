import json
from glob import glob

for output_filename in glob('out/**.json'):
    with open(output_filename, 'r') as r:
        output_json = json.load(r)
        print(output_json['scores'][-1])
