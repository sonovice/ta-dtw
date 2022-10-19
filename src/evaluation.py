import os
import math

import librosa
import numpy as np
from addict import Addict
from tqdm import tqdm

import data
import feature
from dtw import dtw, ta_dtw, compute_cost_matrix, compute_accumulated_cost_matrix, backtracking

test_intervals = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]


def evaluate(rec_id, has_drift=False, feature_type='chroma', alignment='dtw', metric='cosine', transposition_penalty=6.5, hop_length=1024):
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
        alignment_path = np.asarray(ta_dtw(score_features, audio_features, transposition_penalty=transposition_penalty))
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
    metric = kwargs.get('metric') or 'cosine'

    rec_ids = data.get_ids()
    rec_ids = [rec_id for rec_id in rec_ids if os.path.exists(os.path.join('data', 'drift', f'{rec_id}.wav'))]

    # rec_ids = ['2529']

    result_dir = f"results/{feature_type}_{alignment}_{'drift' if has_drift else 'original'}"
    os.makedirs(result_dir, exist_ok=True)

    for rec_id in tqdm(rec_ids, desc=f"{feature_type}_{alignment}_{'drift' if has_drift else 'original'}"):
        target_path = os.path.join(result_dir, f"{rec_id}.npy")
        if os.path.exists(target_path):
            continue

        audio = data.extract_audio(rec_id)
        length_seconds = len(audio) / 44100

        try:
            results = evaluate(rec_id, has_drift=has_drift, alignment=alignment, feature_type=feature_type, transposition_penalty=6.5, metric=metric, hop_length=hop_length)
            np.save(target_path, results)
            tqdm.write(f'================== {rec_id} (len: {round(length_seconds)})==================')
            tqdm.write(f"Features: {feature_type}, DTW: {alignment}, Audio: {'drift' if has_drift else 'original'}")
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
    intervals = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
    hop_length = 1024
    max_length_seconds = 300

    rec_ids = data.get_ids()
    rec_ids = [rec_id for rec_id in rec_ids if os.path.exists(os.path.join('data', 'drift', f'{rec_id}.wav'))]

    results = Addict()

    for rec_id in tqdm(rec_ids, desc="File"):
        audio_path = os.path.join('data', 'drift', f'{rec_id}.wav')

        audio = data.extract_audio(rec_id)
        len_audio = audio.shape[0] // 2
        length_seconds = len(audio) / 44100
        if length_seconds >= max_length_seconds:
            continue

        tqdm.write(f'=== {rec_id} (length: {round(length_seconds)}s) =============================')
        for interval in intervals:
            interval_str = str(interval)
            tqdm.write(f"  Interval: {interval}")

            audio, sr = librosa.load(audio_path, sr=22050)
            audio = audio[:len_audio]
            audio_features = feature.audio_to_pcp(audio, hop_length=hop_length)
            del audio
            length = audio_features.shape[1]

            score_features = feature.pitch_to_chroma(data.extract_score_pitches(rec_id, length))
            ground_truth = data.extract_ground_truth(rec_id, length)

            cost_matrix = compute_cost_matrix(score_features, audio_features)

            for tp in np.arange(1, 11, step=1):
                tp_str = str(tp)
                alignment_path = np.asarray(backtracking(compute_accumulated_cost_matrix(cost_matrix, transposition_penalty=tp)))
                onsets_audio = np.interp(ground_truth[1], sorted(alignment_path[:, 0]), sorted(alignment_path[:, 1]))
                diff = np.abs(onsets_audio[10:-10] - ground_truth[0][10:-10])
                diff_s = diff * (hop_length / 22050)
                count = np.count_nonzero(diff_s <= interval)
                score = count / len(diff_s)

                results[rec_id][interval_str][tp_str] = score
                tqdm.write(f"    {tp}: {score}")

        with open("eval_transposition_penalty.json", "w") as fp:
            json.dump(results, fp)


def run_experiments():
    experiments = [
        dict(has_drift=True, feature_type='chroma', alignment='dtw'),
        dict(has_drift=True, feature_type='chroma', alignment='ta-dtw'),
        dict(has_drift=True, feature_type='pcp', alignment='ta-dtw'),
        dict(has_drift=False, feature_type='chroma', alignment='dtw'),
        dict(has_drift=False, feature_type='chroma', alignment='ta-dtw'),
        dict(has_drift=False, feature_type='pcp', alignment='dtw'),
        dict(has_drift=False, feature_type='pcp', alignment='ta-dtw'),
        # dict(has_drift=False, feature_type='none', alignment='none'),
    ]

    ctx = mp.get_context('spawn')
    pool = ctx.Pool(5)

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
