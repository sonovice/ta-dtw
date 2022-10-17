import os
import math

import librosa.display
import numpy as np
from addict import Addict
from tqdm import tqdm

import data
import feature
from dtw import dtw, ta_dtw, compute_cost_matrix, compute_accumulated_cost_matrix, backtracking

test_intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5, 10]


def evaluate(rec_id, has_drift=False, feature_type='chroma', alignment='dtw', metric='cosine', transposition_penalty=5.0, hop_length=1024):
    type_string = 'drift' if has_drift else 'original'

    # Load audio
    audio, sr = librosa.load(os.path.join('data', type_string, f'{rec_id}.wav'))

    # Compute feature_type
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
    ground_truth = data.extract_ground_truth(rec_id, length)

    # Compute DTW alignment
    if alignment == 'dtw':
        alignment_path = np.asarray(dtw(score_features, audio_features, metric=metric))
    elif alignment == 'ta-dtw':
        alignment_path = np.asarray(ta_dtw(score_features, audio_features, metric=metric, transposition_penalty=transposition_penalty))
    elif alignment == 'none':
        alignment_path = np.arange(start=length - 1, stop=-1, step=-1)
        alignment_path = np.stack([alignment_path, alignment_path], axis=1)

    onsets_audio = np.interp(ground_truth[1], sorted(alignment_path[:, 0]), sorted(alignment_path[:, 1]))
    diff = np.abs(onsets_audio[10:-10] - ground_truth[0][10:-10])
    diff_s = diff * (hop_length / 22050)

    results = []
    for test in test_intervals:
        count = np.count_nonzero(diff_s <= test)
        results.append((count, len(diff_s)))

    return results


def batch(kwargs):
    has_drift, feature_type, alignment = kwargs['has_drift'], kwargs['feature_type'], kwargs['alignment']

    rec_ids = data.get_ids()
    rec_ids = ['1728']
    max_length_seconds = 600

    result_dir = f"results/{feature_type}_{alignment}_{'drift' if has_drift else 'original'}"
    os.makedirs(result_dir, exist_ok=True)

    for rec_id in tqdm(rec_ids, desc=f"{feature_type}_{alignment}_{'drift' if has_drift else 'original'}"):
        target_path = os.path.join(result_dir, f"{rec_id}.npy")

        audio = data.extract_audio(rec_id)
        length_seconds = len(audio) / 44100
        if length_seconds >= max_length_seconds:
            continue
        if os.path.exists(target_path):
            continue

        try:
            results = evaluate(rec_id, has_drift=has_drift, alignment=alignment, feature_type=feature_type)
            # np.save(target_path, results)
            tqdm.write(f'================== {rec_id} (len: {round(length_seconds)})==================')
            tqdm.write(f"Features: {feature_type}, DTW: {alignment}, Audio: {'drift' if has_drift else 'original'}")
            for interval, result in zip(test_intervals, results):
                count, len_diff = result
                percent = count / len_diff * 100
                tqdm.write(f'{interval:.2f}s:\t{percent:.2f}%')

            tqdm.write('')
        except MemoryError:
            max_length_seconds = length_seconds
            tqdm.write(f'Memory overflow. Length: {int(length_seconds)}s. Skipping...')


def eval_metrics(rec_id, has_drift, feature_type, alignment, transposition_penalty, hop_length):
    results_dict = Addict()

    for test_interval, (count, len_diff) in zip(test_intervals,
                                                evaluate(rec_id, has_drift=has_drift, feature_type=feature_type, alignment=alignment, transposition_penalty=transposition_penalty,
                                                         hop_length=hop_length, metric='cosine')):
        results_dict['cosine'][test_interval] = count / len_diff * 100
    for test_interval, (count, len_diff) in zip(test_intervals,
                                                evaluate(rec_id, has_drift=has_drift, feature_type=feature_type, alignment=alignment, transposition_penalty=transposition_penalty,
                                                         hop_length=hop_length, metric='euclidean')):
        results_dict['euclidean'][test_interval] = count / len_diff * 100
    for test_interval, (count, len_diff) in zip(test_intervals,
                                                evaluate(rec_id, has_drift=has_drift, feature_type=feature_type, alignment=alignment, transposition_penalty=transposition_penalty,
                                                         hop_length=hop_length, metric='cityblock')):
        results_dict['cityblock'][test_interval] = count / len_diff * 100

    return rec_id, results_dict


def run_eval_metrics():
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(6)

    metrics_path = 'eval_metrics.json'
    if not os.path.exists(metrics_path):
        results = Addict()
        rec_ids = data.get_ids()
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
    # intervals = list(json_data[list(json_data.keys())[0]][metrics[0]].keys())
    intervals = ['0.15', '0.2', '0.25', '0.3', '0.4', '0.5', '1']
    ids = list(json_data.keys())

    cumulator = Addict()

    for metric in tqdm(metrics):
        for interval in intervals:
            if interval not in cumulator[metric]:
                cumulator[metric][interval] = []
                for id in ids:
                    cumulator[metric][interval].append(json_data[id][metric][interval])

    with open("eval_metrics_cumulated.json", "w") as fp:
        json.dump(cumulator, fp)


def run_eval_transposition_penalty():
    intervals = [0.25, 0.5]
    hop_length = 1024
    rec_id = '1727'

    for interval in tqdm(intervals, desc='Interval'):
        tqdm.write(f"\nInterval: {interval}")

        audio, sr = librosa.load(os.path.join('data', 'drift', f'{rec_id}.wav'))
        audio_features = feature.audio_to_pcp(audio, hop_length=hop_length)
        del audio
        length = audio_features.shape[1]

        score_features = feature.pitch_to_chroma(data.extract_score_pitches(rec_id, length))
        ground_truth = data.extract_ground_truth(rec_id, length)

        cost_matrix = compute_cost_matrix(score_features, audio_features, metric='cosine')

        for tp in tqdm(np.arange(1, 15, step=0.5), desc="TP value"):
            alignment_path = np.asarray(backtracking(compute_accumulated_cost_matrix(cost_matrix, transposition_penalty=tp)))
            onsets_audio = np.interp(ground_truth[1], sorted(alignment_path[:, 0]), sorted(alignment_path[:, 1]))
            diff = np.abs(onsets_audio[10:-10] - ground_truth[0][10:-10])
            diff_s = diff * (hop_length / 22050)
            count = np.count_nonzero(diff_s <= interval)
            score = count / len(diff_s)

            tqdm.write(f"{tp}: {score}")


def run_experiments():
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(6)

    experiments = [
        # dict(has_drift=False, feature_type='chroma', alignment='dtw'),
        # dict(has_drift=True, feature_type='chroma', alignment='dtw'),
        dict(has_drift=True, feature_type='pcp', alignment='ta-dtw'),
        dict(has_drift=False, feature_type='pcp', alignment='ta-dtw'),
        # dict(has_drift=False, feature_type='pcp', alignment='dtw'),
        # dict(has_drift=False, feature_type='none', alignment='none'),
    ]

    for _ in tqdm(pool.imap_unordered(batch, experiments)):
        pass


if __name__ == '__main__':
    import json
    import multiprocessing as mp
    import os
    from functools import partial

    # run_eval_metrics()  # make grouped violin plot like here: https://www.python-graph-gallery.com/54-grouped-violinplot
    run_eval_transposition_penalty()
    # run_experiments()
