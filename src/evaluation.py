import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from addict import Addict
from tqdm import tqdm

import dataset
import dtw

HOP_LENGTH = 1024
SAMPLE_RATE = 22050
INTERVALS = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]

matplotlib.use('tkagg')
sns.set_theme(style="darkgrid")


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


def eval_metrics_worker(args):
    rec_id, metric = args
    results = evaluate(rec_id, has_drift=False, feature_type='chroma', is_transposition_aware=False, metric=metric)
    return metric, results


def run_eval_metrics(force=False, show=False):
    results_path = '../results/metrics.json'

    if force:
        os.remove(results_path)

    if not Path(results_path).exists():
        rec_ids = dataset.features.keys()
        metrics = ['euclidean', 'cosine', 'cityblock']
        combinations = list(itertools.product(rec_ids, metrics))

        metrics_results = Addict()

        ctx = mp.get_context('spawn')
        pool = ctx.Pool()
        for metric, results in tqdm(pool.imap_unordered(eval_metrics_worker, combinations), total=len(combinations)):
            if metric not in metrics_results:
                metrics_results[metric].num_onsets = 0
            metrics_results[metric].num_onsets += results.num_onsets

            for interval in INTERVALS:
                if interval not in metrics_results[metric].matches_in_interval:
                    metrics_results[metric].matches_in_interval[interval] = 0
                metrics_results[metric].matches_in_interval[interval] += results.matches_in_interval[interval]

        data = Addict()
        for metric in metrics:
            intervals = metrics_results[metric].matches_in_interval.keys()
            for interval in intervals:
                score = (metrics_results[metric].matches_in_interval[interval] / metrics_results[metric].num_onsets) * 100
                data[metric][interval] = score
        with open(results_path, 'w') as fp:
            json.dump(data, fp)
    else:
        with open(results_path, 'r') as fp:
            data = json.load(fp)

    if show:
        fig, ax = plt.subplots(constrained_layout=True)
        for metric in data.keys():
            intervals = list(data[metric].keys())
            x = [f'≤{i}s' for i in intervals]
            y = [data[metric][interval] for interval in intervals]
            ax.plot(x, y, '--o', label=metric)
            ax.set_ylim(86, 100)
        plt.legend()
        plt.show()


def run_eval_transposition_penalty(force=False):
    results_path = '../results/transposition_penalty.json'

    if force:
        os.remove(results_path)

    if not Path(results_path).exists():
        pass


def experiment_worker(args):
    return evaluate(*args), args


def run_experiments():
    results_path = Path('../results/experiments')
    results_path.mkdir(parents=True, exist_ok=True)

    rec_ids = list(dataset.features.keys())

    # @formatter:off
    experiments_all = list(itertools.product(
        rec_ids,
        [True],          # has_drift
        ['pcp'],         # feature_type
        [True],          # is_transposition_aware
        ['cosine'],      # metric
        range(1, 52, 5)  # transposition_penalty
    ))
    # @formatter:on

    # Remove already done experiments
    experiments = []
    for exp in experiments_all:
        rec_id, has_drift, feature_type, is_transposition_aware, metric, transposition_penalty = exp
        file_name = f"{rec_id}_drift={has_drift}_feature={feature_type}_algo={'ta-dtw' if is_transposition_aware else 'dtw'}_metric={metric}_tp={transposition_penalty}.json"
        if not (results_path / file_name).exists():
            experiments.append(exp)

    with mp.get_context('spawn').Pool(6) as pool:
        for results, args in tqdm(pool.imap_unordered(experiment_worker, experiments), total=len(experiments)):
            rec_id, has_drift, feature_type, is_transposition_aware, metric, transposition_penalty = args
            file_name = f"{rec_id}_drift={has_drift}_feature={feature_type}_algo={'ta-dtw' if is_transposition_aware else 'dtw'}_metric={metric}_tp={transposition_penalty}.json"
            with open(results_path / file_name, 'w') as fp:
                json.dump(results, fp)


if __name__ == '__main__':
    Path('../results').mkdir(exist_ok=True)

    run_eval_metrics()
    run_eval_transposition_penalty()
    run_experiments()
