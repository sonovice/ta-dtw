import math
import multiprocessing as mp
from pathlib import Path

import librosa
import numpy as np
from addict import Addict
from tqdm import tqdm

import feature

BLACKLIST = ['1824', '1932', '2131', '2296', '2350']  # files with faulty ground truth
HOP_LENGTH = 1024
NOTE_VALUES_DICT = {
    'Dotted Eighth': (1 / 8) * 1.5,
    'Tied Quarter-Thirty Second': (1 / 4) + 0.125,
    'Quarter': (1 / 4),
    'Whole': (1 / 4),
    'Dotted Quarter': (1 / 4) * 1.5,
    'Half': (1 / 2),
    'Unknown': 0,
    'Triplet Sixteenth': (1 / 16) * (2 / 3),
    'Dotted Sixteenth': (1 / 16) * 1.5,
    'Sixteenth': (1 / 16),
    'Thirty Second': (1 / 32),
    'Tied Quarter-Sixteenth': (1 / 4) + (1 / 16),
    'Triplet Sixty Fourth': (1 / 64) * (2 / 3),
    'Eighth': (1 / 8),
    'Sixty Fourth': (1 / 64),
    'Triplet Thirty Second': (1 / 32) * (2 / 3),
    'Triplet': (1 / 8) * (2 / 3),
    'Dotted Half': (1 / 2) * 1.5
}

dataset = np.load('../data/musicnet.npz', mmap_mode='r', encoding='latin1', allow_pickle=True)


def extract_score_pitches(rec_id, frames):
    audio, labels = dataset[rec_id]

    n_measures = 0
    notes = []
    for label in labels:
        (start, end, (instrument, note, measure, beat, note_value)) = label
        n_measures = max(n_measures, measure)

        notes.append(
            {'pitch': note,
             'date': measure + float(f'{beat:.6f}'),
             'dur': NOTE_VALUES_DICT[note_value]
             }
        )

    pitches = np.zeros((12 * 11, frames), dtype=np.float64)
    for note in notes:
        begin = int(math.floor(note['date'] / n_measures * frames))
        if note['pitch'] is not None:
            end = int(math.ceil((note['date'] + note['dur']) / n_measures * frames))
            for c in range(begin, end):
                try:
                    pitches[note['pitch'], c] += 1
                except IndexError:
                    pass

    pitches = np.log(pitches + 1)[24:24 + 12 * 8, :]

    return pitches


def compute_features(file_path):
    rec_id = file_path.stem
    if rec_id in BLACKLIST:
        return

    audio, _ = dataset[rec_id]
    audio_length = audio.shape[0] // 2

    features = Addict()
    for type_string in ['drift', 'original']:
        audio, sr = librosa.load(f'../data/{type_string}/{rec_id}.wav', sr=22050)
        audio = audio[:audio_length]

        features_chroma = feature.audio_to_chroma(audio, hop_length=HOP_LENGTH)
        features_pcp = feature.audio_to_pcp(audio, hop_length=HOP_LENGTH)
        features_score = feature.pitch_to_chroma(extract_score_pitches(rec_id, features_chroma.shape[1]))

        features[rec_id][type_string]['chroma'] = features_chroma
        features[rec_id][type_string]['pcp'] = features_pcp
        features[rec_id]['score'] = features_score

    return features


if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(16)

    paths = list(Path('../data/drift').glob('*.wav'))
    features_all = Addict()

    for features in tqdm(pool.imap_unordered(compute_features, paths), total=len(paths)):
        if features is not None:
            features_all.update(features)

    np.save('../data/features.npy', dict(features_all))
