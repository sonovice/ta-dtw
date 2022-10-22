import numpy as np

dataset_path = '../data/musicnet.npz'
dataset = np.load(dataset_path, mmap_mode='r', encoding='latin1', allow_pickle=True)
features: dict = np.load('../data/musicnet_features.npy', allow_pickle=True).item()


def extract_ground_truth(rec_id, max_frame):
    audio, labels = dataset[rec_id]

    length = audio.shape[0]
    n_measures = 0
    audio_onsets = []
    score_onsets = []

    for label in sorted(labels):
        (start, end, (instrument, note, measure, beat, note_value)) = label
        n_measures = max(n_measures, measure)

    for label in sorted(labels):
        (start, end, (instrument, note, measure, beat, note_value)) = label
        audio_onsets.append(start / length * max_frame)
        score_onsets.append((measure + float(f'{beat:.6f}')) / n_measures * max_frame)

    return audio_onsets, score_onsets
