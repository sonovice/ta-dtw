import glob

import numpy as np

test_intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1]
test_sets = [
    'std_std_pitched',
    'std_std_unpitched',
    'ta_ta_pitched',
    'ta_ta_unpitched',
    'ta_std_unpitched'
]

for test_set in test_sets:
    files = glob.glob(f'../data/results/*_{test_set}.npy')
    if len(files) == 0:
        continue

    results = np.zeros((11, 2))
    print(test_set)

    for file in files:
        results += np.load(file)
    for i, test in enumerate(test_intervals):
        print(f'{test:.2f}s:\t{results[i, 0] / results[i, 1] * 100:.2f}%', )

    print()
