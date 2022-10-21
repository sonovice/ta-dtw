import os
import math
import shutil
import statistics
from collections import defaultdict
from glob import glob

import librosa
import numpy as np
from addict import Addict
from tqdm import tqdm

import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

import data
import feature
from dtw import dtw, ta_dtw, compute_cost_matrix, compute_accumulated_cost_matrix, backtracking

test_intervals = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]


def evaluate(rec_id, has_drift=False, feature_type='chroma', alignment='dtw', metric='cityblock', transposition_penalty=6.5, hop_length=1024):
    type_string = 'drift' if has_drift else 'original'

    # Load audio
    audio = data.extract_audio(rec_id)
    audio_length = audio.shape[0] // 2
    audio, sr = librosa.load(os.path.join('data', type_string, f'{rec_id}.wav'), sr=22050)
    audio = audio[:audio_length]

    # Compute features
    if feature_type == 'pcp':
        audio_features = feature.audio_to_pcp(audio, hop_length=hop_length)
        length = audio_features.shape[1]
    elif feature_type == 'chroma':
        audio_features = feature.audio_to_chroma(audio, hop_length=hop_length)
        length = audio_features.shape[1]
    elif feature_type == 'none':
        length = math.ceil(audio.shape[0] / hop_length)

    del audio

    # Load ground truth
    score_features = feature.pitch_to_chroma(data.extract_score_pitches(rec_id, length))
    gt_audio_onsets, gt_score_onsets = data.extract_ground_truth(rec_id, length)

    # Compute alignment
    if alignment == 'dtw':
        alignment_path = np.asarray(dtw(score_features, audio_features, metric=metric))
    elif alignment == 'ta-dtw':
        alignment_path = np.asarray(ta_dtw(score_features, audio_features, transposition_penalty=transposition_penalty, metric=metric))
    elif alignment == 'none':
        alignment_path = np.arange(start=length - 1, stop=-1, step=-1)
        alignment_path = np.stack([alignment_path, alignment_path], axis=1)

    # Compute scores
    alignment_path = np.flip(alignment_path, axis=0)
    audio_onsets = np.interp(gt_score_onsets, alignment_path[:, 0], alignment_path[:, 1])
    diff = np.abs(audio_onsets[10:-10] - gt_audio_onsets[10:-10])
    diff_s = diff * (hop_length / 22050)

    results = []
    for test in test_intervals:
        count = np.count_nonzero(diff_s <= test)
        results.append((count, len(diff_s)))

    return results


def batch(kwargs):
    hop_length = 1024
    has_drift, feature_type, alignment = kwargs['has_drift'], kwargs['feature_type'], kwargs['alignment']
    transposition_penalty = kwargs.get('transposition_penalty') or 6.5
    metric = kwargs.get('metric') or 'euclidean'
    path = kwargs.get('path') or 'results'

    rec_ids = data.get_ids()
    rec_ids = [rec_id for rec_id in rec_ids if os.path.exists(os.path.join('data', 'drift', f'{rec_id}.wav'))]

    # rec_ids = ['2529']

    result_dir = f"{path}/{feature_type}_{alignment}_{'drift' if has_drift else 'original'}"
    os.makedirs(result_dir, exist_ok=True)

    for rec_id in tqdm(rec_ids, desc=f"{feature_type}_{alignment}_{'drift' if has_drift else 'original'}"):
        target_path = os.path.join(result_dir, f"{rec_id}.npy")
        if os.path.exists(target_path):
            continue

        audio = data.extract_audio(rec_id)
        length_seconds = len(audio) / 44100

        try:
            results = evaluate(rec_id, has_drift=has_drift, alignment=alignment, feature_type=feature_type, transposition_penalty=transposition_penalty, metric=metric, hop_length=hop_length)
            np.save(target_path, results)
            tqdm.write(f'================== {rec_id} (len: {round(length_seconds)})==================')
            tqdm.write(f"Features: {feature_type}, Algorithm: {alignment}, Audio: {'drift' if has_drift else 'original'}")
            for interval, result in zip(test_intervals, results):
                count, len_diff = result
                percent = count / len_diff * 100
                tqdm.write(f'{interval:.2f}s:\t{percent:.2f}%')

            tqdm.write('')
        except MemoryError:
            tqdm.write(f'Memory overflow. Length: {int(length_seconds)}s. Skipping...')


def eval_metrics(rec_id, has_drift, feature_type, alignment, transposition_penalty, hop_length):
    results_dict = Addict()

    for test_interval, (count, len_diff) in zip(test_intervals,
                                                evaluate(rec_id, has_drift=has_drift, feature_type=feature_type, alignment=alignment, transposition_penalty=transposition_penalty,
                                                         hop_length=hop_length, metric='cosine')):
        results_dict['cosine'][test_interval] = [count, len_diff]
    for test_interval, (count, len_diff) in zip(test_intervals,
                                                evaluate(rec_id, has_drift=has_drift, feature_type=feature_type, alignment=alignment, transposition_penalty=transposition_penalty,
                                                         hop_length=hop_length, metric='euclidean')):
        results_dict['euclidean'][test_interval] = [count, len_diff]
    for test_interval, (count, len_diff) in zip(test_intervals,
                                                evaluate(rec_id, has_drift=has_drift, feature_type=feature_type, alignment=alignment, transposition_penalty=transposition_penalty,
                                                         hop_length=hop_length, metric='cityblock')):
        results_dict['cityblock'][test_interval] = [count, len_diff]

    return rec_id, results_dict


def run_eval_metrics():
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(6)

    metrics_path = 'eval_metrics.json'
    if not os.path.exists(metrics_path):
        results = Addict()
        rec_ids = data.get_ids()
        rec_ids = [rec_id for rec_id in rec_ids if os.path.exists(os.path.join('data', 'drift', f'{rec_id}.wav'))]
        for id, results_dict in tqdm(
                pool.imap_unordered(partial(eval_metrics, has_drift=False, feature_type='chroma', alignment='dtw', transposition_penalty=1, hop_length=1024), rec_ids),
                total=len(rec_ids)):
            results[id] = results_dict
            # tqdm.write(f'=== {id} ===  {results_dict.__repr__()}')

        with open("eval_metrics.json", "w") as fp:
            json.dump(results, fp)

    with open("eval_metrics.json") as fp:
        json_data = json.load(fp)

    metrics = list(json_data[list(json_data.keys())[0]].keys())
    metrics = ['cosine', 'euclidean']
    # intervals = list(json_data[list(json_data.keys())[0]][metrics[0]].keys())
    intervals = ['0.15', '0.2', '0.25', '0.3', '0.4', '0.5', '1.0']
    ids = list(json_data.keys())

    cumulator = Addict()

    for metric in metrics:
        for interval in intervals:
            if interval not in cumulator[metric]:
                cumulator[metric][interval] = []
            for id in ids:
                cumulator[metric][interval].append(json_data[id][metric][interval])

            correct = sum([vals[0] for vals in cumulator[metric][interval]])
            all = sum([vals[1] for vals in cumulator[metric][interval]])
            cumulator[metric][interval] = correct / all

    with open("eval_metrics_cumulated.json", "w") as fp:
        json.dump(cumulator, fp)

    for metric in metrics:
        plt.scatter(list(cumulator[metric].keys()), list(cumulator[metric].values()), label=metric)
        plt.plot(list(cumulator[metric].keys()), list(cumulator[metric].values()), '--')
    plt.xlabel("Intervals [s]")
    plt.ylabel("Scores")
    plt.legend()
    plt.show()


def eval_transposition_penalty(rec_id, intervals, hop_length, max_length_seconds):
    audio_path = os.path.join('data', 'drift', f'{rec_id}.wav')

    audio = data.extract_audio(rec_id)
    len_audio = audio.shape[0] // 2
    length_seconds = len(audio) / 44100
    if length_seconds >= max_length_seconds:
        return {}

    # tqdm.write(f'=== {rec_id} (length: {round(length_seconds)}s) =============================')

    results = Addict()
    for interval in intervals:
        interval_str = str(interval)
        # tqdm.write(f"  Interval: {interval}")

        audio, sr = librosa.load(audio_path, sr=22050)
        audio = audio[:len_audio]
        audio_features = feature.audio_to_pcp(audio, hop_length=hop_length)
        del audio
        length = audio_features.shape[1]

        score_features = feature.pitch_to_chroma(data.extract_score_pitches(rec_id, length))
        ground_truth = data.extract_ground_truth(rec_id, length)

        cost_matrix = compute_cost_matrix(score_features, audio_features, metric='euclidean')

        for tp in np.arange(3.5, 10.5, step=0.5):
            tp_str = str(tp)
            alignment_path = np.asarray(backtracking(compute_accumulated_cost_matrix(cost_matrix, transposition_penalty=tp)))
            onsets_audio = np.interp(ground_truth[+1], sorted(alignment_path[:, 0]), sorted(alignment_path[:, 1]))
            diff = np.abs(onsets_audio[10:-10] - ground_truth[0][10:-10])
            diff_s = diff * (hop_length / 22050)
            num_correct = np.count_nonzero(diff_s <= interval)
            num_all = len(diff_s)

            results[rec_id][interval_str][tp_str] = [num_correct, num_all]

            # tqdm.write(f"    {tp}: {num_correct / num_all}")

    return results


def run_eval_transposition_penalty():
    # intervals = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
    intervals = [0.3]
    hop_length = 1024
    max_length_seconds = 300

    path = "eval_transposition_penalty.json"

    if os.path.exists(path):
        with open(path) as fp:
            results_cumulated = json.load(fp)
        results_cumulated = Addict(results_cumulated)
    else:
        results_cumulated = Addict()

    rec_ids = data.get_ids()
    rec_ids = [rec_id for rec_id in rec_ids if os.path.exists(os.path.join('data', 'drift', f'{rec_id}.wav'))]
    rec_ids = rec_ids[:4]
    # rec_ids = [rec_id for rec_id in rec_ids if rec_id not in results_cumulated.keys()]

    print(len(rec_ids))

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(min(14, len(rec_ids)))

    for results in tqdm(pool.imap_unordered(partial(eval_transposition_penalty, intervals=intervals, hop_length=hop_length, max_length_seconds=max_length_seconds), rec_ids),
                        total=len(rec_ids)):
        results_cumulated.update(results)
        with open(path, "w") as fp:
            json.dump(results_cumulated, fp)

    with open(path) as fp:
        results_cumulated = json.load(fp)

    results = Addict()
    for values in results_cumulated.values():
        for interval, values in values.items():
            for tp, values in values.items():
                if tp not in results[interval]:
                    results[interval][tp]['num_correct'] = 0
                    results[interval][tp]['num_all'] = 0
                num_correct, num_all = values
                results[interval][tp]['num_correct'] += num_correct
                results[interval][tp]['num_all'] += num_all

    final_results = Addict()
    for interval, values in results.items():
        for tp in values.keys():
            final_results[interval][tp] = results[interval][tp]['num_correct'] / results[interval][tp]['num_all']

    with open("eval_transposition_penalty_cumulated.json", "w") as fp:
        json.dump(final_results, fp)

    # for tp, values in final_results.items():
    #     intervals = list(values.keys())
    #     scores = list(values.values())
    #     plt.scatter(intervals, scores, label=str(tp))
    #     plt.plot(intervals, scores, '--')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # x = sorted([float(v) for v in final_results.keys()])
    # y = [v['0.3'] for v in final_results.values()]
    # plt.scatter(x, y, label=str(tp))
    # plt.plot(x, y, '--')
    # plt.tight_layout()
    # plt.show()


def compute_stats_experiments(path):
    print(f'=== {path} ===')

    blacklist = []
    for file in glob(f'{path}/*.npy'):
        basename = os.path.basename(file)
        c, a = np.load(file)[5]
        if c/a == 0:
            blacklist.append(basename)
    print(f"Blacklist: {blacklist}\n")

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle(path)
    axs[0].set_title('drift')
    axs[1].set_title('original')
    axs[0].set_ylim([0, 100])
    axs[1].set_ylim([0, 100])

    dirs = sorted(glob(f'{path}/*'))
    for dir in dirs:
        nums_correct = defaultdict(int)
        nums_all = defaultdict(int)
        files = glob(f'{dir}/*.npy')
        for file in files:
            basename = os.path.basename(file)
            if basename in blacklist:
                continue
            data = np.load(file)
            for i, interval in enumerate(test_intervals):
                nums_correct[interval] += data[i, 0]
                nums_all[interval] += data[i, 1]

        dirname = dir.split('/')[-1]
        print(dirname)
        x = [f'≤{k}s' for k in nums_correct.keys()]
        y = []
        for interval in nums_correct.keys():
            num_correct = nums_correct[interval]
            num_all = nums_all[interval]
            score = num_correct/num_all
            y.append(score * 100)
            print(f'{interval}: {score}')

        ax = 0 if dirname.endswith("drift") else 1
        line = '--' if 'ta-dtw' in dirname else ':'
        line += 'o' if 'pcp' in dirname else 'x'
        axs[ax].plot(x, y, line, label=dirname)

        print()

    axs[0].legend()
    axs[1].legend()
    # plt.tight_layout()
    plt.show()


def run_experiments(path, transposition_penalty, metric='euclidean'):
    # @formatter:off
    experiments = [
        dict(path=path, has_drift=False, feature_type='chroma', alignment='dtw',    metric=metric, transposition_penalty=transposition_penalty),
        dict(path=path, has_drift=False, feature_type='chroma', alignment='ta-dtw', metric=metric, transposition_penalty=transposition_penalty),
        dict(path=path, has_drift=False, feature_type='pcp',    alignment='dtw',    metric=metric, transposition_penalty=transposition_penalty),
        dict(path=path, has_drift=False, feature_type='pcp',    alignment='ta-dtw', metric=metric, transposition_penalty=transposition_penalty),

        dict(path=path, has_drift=True,  feature_type='chroma', alignment='dtw',    metric=metric, transposition_penalty=transposition_penalty),
        dict(path=path, has_drift=True,  feature_type='chroma', alignment='ta-dtw', metric=metric, transposition_penalty=transposition_penalty),
        dict(path=path, has_drift=True,  feature_type='pcp',    alignment='dtw',    metric=metric, transposition_penalty=transposition_penalty),
        dict(path=path, has_drift=True,  feature_type='pcp',    alignment='ta-dtw', metric=metric, transposition_penalty=transposition_penalty),
        # dict(has_drift=False, feature_type='none',   alignment='none'),
    ]
    # @formatter:on

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(len(experiments))

    for _ in tqdm(pool.imap_unordered(batch, experiments)):
        pass


if __name__ == '__main__':
    import json
    import multiprocessing as mp
    import os
    from functools import partial

    # run_eval_metrics()  # make grouped violin plot like here: https://www.python-graph-gallery.com/54-grouped-violinplot
    # run_eval_transposition_penalty()

    # path = 'results_tp=6.5_chroma=stft_euclidean'
    # path = 'results_tp=24_chroma=cqt_euclidean'
    # path = 'results_tp=24_chroma=cqt_cosine'
    # path = 'results_tp=100_chroma=cqt_euclidean'2
    metric = 'euclidean'
    chroma = 'cqt'

    for tp in [6.5, 24, 100, np.inf]:
        path = f'results/results_tp={tp}_chroma={chroma}_{metric}'
        run_experiments(path, tp, metric)



    compute_stats_experiments('results/results_tp=100_chroma=cqt_euclidean_start-t=0')
