import itertools
import json
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


def experiment_worker(args):
    return evaluate(*args), args


def run_experiments():
    results_path = Path(f'../experiments/')
    results_path.mkdir(parents=True, exist_ok=True)
    rec_ids = list(dataset.features.keys())

    experiments_all = []

    experiments_metrics = list(itertools.product(
        rec_ids,
        [False],  # has_drift
        ['chroma'],  # feature_type
        [False],  # is_transposition_aware
        ['cosine', 'euclidean', 'cityblock'],  # metric
        [1]  # transposition_penalty
    ))
    experiments_all.extend(experiments_metrics)

    experiments_transposition_penalty = list(itertools.product(
        rec_ids,
        [False, True],  # has_drift
        ['pcp'],  # feature_type
        [True],  # is_transposition_aware
        ['cosine'],  # metric
        [1] + list(range(10, 101, 10)) + list(range(5, 30))  + list(np.arange(7, 12.5, 0.5).tolist())  # transposition_penalty
    ))
    experiments_all.extend(experiments_transposition_penalty)

    # experiments_fixed_tp = list(itertools.product(
    #     rec_ids,
    #     [False, True],  # has_drift
    #     ['chroma', 'pcp'],  # feature_type
    #     [False, True],  # is_transposition_aware
    #     ['cosine'],  # metric
    #     [24]  # transposition_penalty
    # ))
    # experiments_all.extend(experiments_fixed_tp)

    # Remove already saved experiments
    experiments = []
    for exp in experiments_all:
        rec_id, has_drift, feature_type, is_transposition_aware, metric, transposition_penalty = exp
        file_name = f"{rec_id}_drift={has_drift}_feature={feature_type}_algo={'ta-dtw' if is_transposition_aware else 'dtw'}_metric={metric}_tp={transposition_penalty}.json"
        if not (results_path / file_name).exists():
            experiments.append(exp)

    # Execute actual experiments and save each results per file
    with mp.get_context('spawn').Pool(6) as pool:
        for results, args in tqdm(pool.imap_unordered(experiment_worker, experiments), total=len(experiments)):
            rec_id, has_drift, feature_type, is_transposition_aware, metric, transposition_penalty = args
            file_name = f"{rec_id}_drift={has_drift}_feature={feature_type}_algo={'ta-dtw' if is_transposition_aware else 'dtw'}_metric={metric}_tp={transposition_penalty}.json"
            with open(results_path / file_name, 'w') as fp:
                json.dump(results, fp)


def index_from_experiments() -> list:
    input_path = Path(f'../experiments/')
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
    index = index_from_experiments()
    index = filter(
        lambda e: all([
            not e.has_drift,
            e.feature_type == 'chroma',
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


def viz_transposition_penalty():
    index = index_from_experiments()
    index = filter(
        lambda e: all([
            e.feature_type == 'pcp',
            e.is_transposition_aware,
            e.metric == 'cosine',
            e.transposition_penalty <= 100 #or e.transposition_penalty == float('inf'),
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
    y_mixed = [(yd+yo)/2 for yd, yo in zip(y_drift, y_original)]
    ax.plot(x, y_drift, ':.', label='With drift')
    ax.plot(x, y_original, ':.', label='Without drift')
    ax.plot(x, y_mixed, '--.', label='Mixed')
    ax.set_title('Transposition penalty factor influence')
    ax.set_xlabel('Transposition penalty factor')
    ax.set_ylabel('Score in percent')
    ax.set_ylim(90, 100)
    ax.legend()


def viz_final_results():
    index = index_from_experiments()
    index = filter(
        lambda e: all([
            e.transposition_penalty == 1,
            e.metric == 'cosine'
        ]),
        index
    )

    results = Addict()
    for entry in index:
        with open(entry.path, 'rb') as fp:
            data = Addict(orjson.loads(fp.read()))
            for interval in INTERVALS:
                if entry.feature_type not in results[str(entry.has_drift)][str(entry.is_transposition_aware)][interval]:
                    results[str(entry.has_drift)][str(entry.is_transposition_aware)][entry.feature_type][interval] = [0, 0]
                results[str(entry.has_drift)][str(entry.is_transposition_aware)][entry.feature_type][interval][0] += data.num_onsets
                results[str(entry.has_drift)][str(entry.is_transposition_aware)][entry.feature_type][interval][1] += data.matches_in_interval[str(interval)]


    # Plot
    fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True, sharey=True)

    # Without drift
    entry = results['False']['False']['chroma']
    intervals = [f'≤{i}s' for i in entry.keys()]
    values = [(matched / num) * 100 for num, matched in list(entry.values())]
    axs[0].plot(intervals, values, ':s', label='DTW, Chroma')

    entry = results['False']['False']['pcp']
    intervals = [f'≤{i}s' for i in entry.keys()]
    values = [(matched / num) * 100 for num, matched in list(entry.values())]
    axs[0].plot(intervals, values, ':o', label='DTW, PCP')

    entry = results['False']['True']['chroma']
    intervals = [f'≤{i}s' for i in entry.keys()]
    values = [(matched / num) * 100 for num, matched in list(entry.values())]
    axs[0].plot(intervals, values, '--s', label='TA-DTW, Chroma')

    entry = results['False']['True']['pcp']
    intervals = [f'≤{i}s' for i in entry.keys()]
    values = [(matched / num) * 100 for num, matched in list(entry.values())]
    axs[0].plot(intervals, values, '--o', label='TA-DTW, PCP')

    axs[0].set_title('Without drift')
    axs[0].set_ylim(0, 100)
    axs[0].legend(loc='lower right')

    # With drift
    entry = results['True']['False']['chroma']
    intervals = [f'≤{i}s' for i in entry.keys()]
    values = [(matched / num) * 100 for num, matched in list(entry.values())]
    axs[1].plot(intervals, values, ':s', label='DTW, Chroma')

    entry = results['True']['False']['pcp']
    intervals = [f'≤{i}s' for i in entry.keys()]
    values = [(matched / num) * 100 for num, matched in list(entry.values())]
    axs[1].plot(intervals, values, ':o', label='DTW, PCP')

    entry = results['True']['True']['chroma']
    intervals = [f'≤{i}s' for i in entry.keys()]
    values = [(matched / num) * 100 for num, matched in list(entry.values())]
    axs[1].plot(intervals, values, '--s', label='TA-DTW, Chroma')

    entry = results['True']['True']['pcp']
    intervals = [f'≤{i}s' for i in entry.keys()]
    values = [(matched / num) * 100 for num, matched in list(entry.values())]
    axs[1].plot(intervals, values, '--o', label='TA-DTW, PCP')

    axs[1].set_title('With drift')
    axs[1].set_ylim(0, 100)
    axs[1].legend(loc='lower right')


if __name__ == '__main__':
    # run_experiments()
    # viz_transposition_penalty()
    # viz_metrics()
    # viz_final_results()

    plt.show()


