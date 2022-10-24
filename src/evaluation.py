import itertools
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import orjson
import seaborn as sns
from addict import Addict
from tqdm import tqdm

import dataset
import dtw

matplotlib.use('tkagg')
sns.set_theme(style="darkgrid")

HOP_LENGTH = 1024
SAMPLE_RATE = 22050
INTERVALS = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]


def evaluate(rec_id, has_drift: bool, feature_type: str, is_transposition_aware: bool, metric: str, transposition_penalty: float = 1,
             intervals: Optional[list[float]] = None) -> Addict:
    intervals = intervals or INTERVALS

    audio_features = dataset.features[rec_id]['drift' if has_drift else 'original'][feature_type]
    score_features = dataset.features[rec_id].score

    alignment_path = dtw.align(
        score_features,
        audio_features,
        is_transposition_aware=is_transposition_aware,
        transposition_penalty=transposition_penalty,
        metric=metric
    )

    # Compute scores
    gt_onsets, gt_score_onsets = dataset.extract_ground_truth(rec_id, audio_features.shape[1])
    computed_onsets = np.interp(gt_score_onsets, alignment_path[:, 0], alignment_path[:, 1])
    deviations = np.abs(computed_onsets[10:-10] - gt_onsets[10:-10]) * (HOP_LENGTH / SAMPLE_RATE)

    results = Addict({
        'num_onsets': len(deviations),
        'matches_in_interval': {interval: np.count_nonzero(deviations <= interval) for interval in intervals}
    })

    return results


def configuration_worker(args):
    return evaluate(*args), args


def run_configurations():
    results_path = Path(f'../configurations/')
    results_path.mkdir(parents=True, exist_ok=True)
    rec_ids = list(dataset.features.keys())

    configurations_all = []

    configurations_metrics = list(itertools.product(
        rec_ids,
        [False],  # has_drift
        ['pcp'],  # feature_type
        [False],  # is_transposition_aware
        ['cosine', 'euclidean', 'cityblock'],  # metric
        [1]  # transposition_penalty
    ))
    configurations_all.extend(configurations_metrics)

    configurations_transposition_penalty = list(itertools.product(
        rec_ids,
        [False, True],  # has_drift
        ['pcp'],  # feature_type
        [True],  # is_transposition_aware
        ['euclidean'],  # metric
        [1, 10, 20] + list(range(1, 11)) + [6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5]  # transposition_penalty
    ))
    configurations_all.extend(configurations_transposition_penalty)

    configurations_fixed_tp = list(itertools.product(
        rec_ids,
        [False, True],  # has_drift
        ['chroma', 'pcp'],  # feature_type
        [False, True],  # is_transposition_aware
        ['euclidean'],  # metric
        [6.7, 8.0, 8.1]  # transposition_penalty
    ))
    configurations_all.extend(configurations_fixed_tp)

    # Remove already saved configurations
    configurations = []
    for config in configurations_all:
        rec_id, has_drift, feature_type, is_transposition_aware, metric, transposition_penalty = config
        file_name = f"{rec_id}_drift={has_drift}_feature={feature_type}_algo={'ta-dtw' if is_transposition_aware else 'dtw'}_metric={metric}_tp={transposition_penalty}.json"
        if not (results_path / file_name).exists():
            configurations.append(config)

    # Execute actual configurations and save each results per file
    with mp.get_context('spawn').Pool(12) as pool:
        for results, args in tqdm(pool.imap_unordered(configuration_worker, configurations), total=len(configurations)):
            rec_id, has_drift, feature_type, is_transposition_aware, metric, transposition_penalty = args
            file_name = f"{rec_id}_drift={has_drift}_feature={feature_type}_algo={'ta-dtw' if is_transposition_aware else 'dtw'}_metric={metric}_tp={transposition_penalty}.json"
            with open(results_path / file_name, 'wb') as fp:
                fp.write(orjson.dumps(results))


def index_from_configurations() -> list:
    input_path = Path(f'../configurations/')
    index = []
    for path in input_path.glob('*.json'):
        parts = str(path.stem).split('_')
        parts = [p.split('=')[1] if '=' in p else p for p in parts]
        index.append(Addict({
            'has_drift': parts[1] == 'True',
            'feature_type': parts[2],
            'is_transposition_aware': parts[3] == 'ta-dtw',
            'metric': parts[4],
            'transposition_penalty': float(parts[5]),
            'path': path
        }))

    return index


def viz_metrics():
    index = index_from_configurations()
    index = filter(
        lambda e: all([
            not e.has_drift,
            e.feature_type == 'pcp',
            not e.is_transposition_aware,
            e.transposition_penalty == 1,
        ]),
        index
    )

    metrics = ['cosine', 'euclidean', 'cityblock']
    results = Addict({
        metric: {
            'num_onsets': 0,
            'matches_in_interval': {interval: 0 for interval in INTERVALS}
        } for metric in metrics
    })
    for entry in index:
        with open(entry.path, 'rb') as fp:
            data = Addict(orjson.loads(fp.read()))
        results[entry.metric].num_onsets += data.num_onsets
        for interval in INTERVALS:
            results[entry.metric][interval] += data.matches_in_interval[str(interval)]

    # Plot
    fig, ax = plt.subplots(constrained_layout=True)
    for m, metric in enumerate(metrics):
        x = [f'≤{i}s' for i in INTERVALS]
        y = [(results[metric][interval] / results[metric].num_onsets) * 100 for interval in INTERVALS]
        ax.plot(x, y, f"--{['o', 's', 'D'][m]}", label=metric)
        ax.set_title('Metrics')
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Percentage of correctly aligned note events')
        ax.set_ylim(86, 100)
    plt.legend(loc='upper left')


def viz_transposition_penalty(metric):
    index = index_from_configurations()
    index = filter(
        lambda e: all([
            e.feature_type == 'pcp',
            e.is_transposition_aware,
            e.metric == metric,
            e.transposition_penalty <= 20,
        ]),
        index
    )

    num_onsets_per_tp_drift = defaultdict(int)
    num_matches_per_tp_drift = defaultdict(int)
    num_onsets_per_tp_original = defaultdict(int)
    num_matches_per_tp_original = defaultdict(int)
    for entry in index:
        with open(entry.path, 'rb') as fp:
            data = Addict(orjson.loads(fp.read()))
        if entry.has_drift:
            num_onsets_per_tp_drift[entry.transposition_penalty] += data.num_onsets
            num_matches_per_tp_drift[entry.transposition_penalty] += data.matches_in_interval['0.3']
        else:
            num_onsets_per_tp_original[entry.transposition_penalty] += data.num_onsets
            num_matches_per_tp_original[entry.transposition_penalty] += data.matches_in_interval['0.3']

    tps = sorted(list(num_onsets_per_tp_drift.keys()))

    # Plot
    fig, ax = plt.subplots(constrained_layout=True)
    x = tps
    y_drift = [(num_matches_per_tp_drift[tp] / num_onsets_per_tp_drift[tp]) * 100 for tp in x]
    y_original = [(num_matches_per_tp_original[tp] / num_onsets_per_tp_original[tp]) * 100 for tp in x]
    y_mixed = [(yd + yo) / 2 for yd, yo in zip(y_drift, y_original)]
    ax.plot(x, y_drift, '--.', label='With drift')
    ax.plot(x, y_original, '--.', label='Without drift')
    ax.plot(x, y_mixed, ':.', label='Mixed')
    ax.set_title('Transposition penalty factor influence')
    ax.set_xlabel('Transposition penalty factor')
    ax.set_ylabel('Score in percent')
    ax.set_ylim(90, 100)
    ax.legend()


def viz_final_results(tp, metric):
    index = index_from_configurations()
    index = filter(
        lambda e: all([
            e.transposition_penalty == tp,
            e.metric == metric
        ]),
        index
    )

    results = Addict()
    for entry in index:
        with open(entry.path, 'rb') as fp:
            data = Addict(orjson.loads(fp.read()))
            for interval in INTERVALS:
                if interval not in results[str(entry.has_drift)][str(entry.is_transposition_aware)][entry.feature_type]:
                    results[str(entry.has_drift)][str(entry.is_transposition_aware)][entry.feature_type][interval] = [0, 0]
                results[str(entry.has_drift)][str(entry.is_transposition_aware)][entry.feature_type][interval][0] += data.num_onsets
                results[str(entry.has_drift)][str(entry.is_transposition_aware)][entry.feature_type][interval][1] += data.matches_in_interval[str(interval)]

    fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True, sharey=True)
    combinations = list(itertools.product(
        ['False', 'True'],  # has drift
        ['False', 'True'],  # is transposition aware
        ['chroma', 'pcp'],  # feature type
    ))

    for has_drift, is_transposition_aware, feature_type in combinations:
        entry = results[has_drift][is_transposition_aware][feature_type]
        has_drift = has_drift == 'True'
        is_transposition_aware = is_transposition_aware == 'True'
        intervals = [f'≤{i}s' for i in entry.keys()]
        values = [(matched / num) * 100 for num, matched in list(entry.values())]

        label = f"{'TA-DTW' if is_transposition_aware else 'DTW'}, {feature_type}"
        style = f"{'--' if is_transposition_aware else ':'}{'s' if feature_type == 'chroma' else 'o'}"
        axs[1 if has_drift else 0].plot(intervals, values, style, label=label)

        print()
        print(f"{'With' if has_drift else 'Without'} drift, {'TA-DTW' if is_transposition_aware else 'DTW'}, {feature_type}")
        for i, v in zip(intervals, values):
            print(f'{i:>6} - {v:.2f}%')

    axs[0].set_title('Without drift')
    axs[0].set_ylim(-10, 110)
    axs[0].legend(loc='lower right')

    axs[1].set_title('With drift')
    axs[1].set_ylim(-10, 110)
    axs[1].legend(loc='lower right')

    fig.suptitle(f"Transposition penalty = {tp}, metric = {metric}")


if __name__ == '__main__':
    run_configurations()

    viz_metrics()
    viz_transposition_penalty('euclidean')
    # viz_final_results(6.7, metric='euclidean')  # best with drift
    viz_final_results(8.0, metric='euclidean')  # best mixed
    # viz_final_results(8.1, metric='euclidean')  # best without drift

    plt.show()
