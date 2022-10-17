import math

import numpy as np

dataset_path = './data/musicnet.npz'
dataset = np.load(dataset_path, mmap_mode='r', encoding='latin1', allow_pickle=True)
rec_ids = sorted(dataset.keys())

dict_note_values = {
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


########################################################################################################################

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


def extract_annotated_pitches(rec_id, sr=22050, hop_length=512):
    audio, labels = dataset[rec_id]
    hop_length = int(hop_length * (44100 / sr))
    pitches = np.zeros((audio.shape[0] // hop_length, 12 * 11))

    for frame in range(pitches.shape[0]):
        labels_in_frame = labels[frame * hop_length]
        for label in labels_in_frame:
            (start, end, (instrument, note, measure, beat, note_value)) = label
            pitches[frame, note] += 1

    pitches = np.log(pitches.T + 1)[24:24 + 12 * 8, :]

    return pitches


def extract_audio(rec_id):
    audio, labels = dataset[rec_id]
    return audio


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
             'dur': dict_note_values[note_value]
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


def get_ids():
    return sorted(dataset.keys())
