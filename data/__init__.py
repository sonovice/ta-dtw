import math
import os

import librosa.display
import numpy as np
from scipy.io import wavfile
#from scipy.spatial.distance import cdist

#from cemfi.database.musicnet import helper
#from cemfi.alignment import dtw

import src.features
import src.util as util

package_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(package_dir, 'musicnet.npz')
dataset = np.load(dataset_path, mmap_mode='r', encoding='latin1')
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

    # Get max values
    length = audio.shape[0]
    n_measures = 0
    audio_onsets = []  # [0]
    score_onsets = []  # [0]
    for label in sorted(labels):
        (start, end, (instrument, note, measure, beat, note_value)) = label
        n_measures = max(n_measures, measure)
    # n_measures += 1

    for label in sorted(labels):
        (start, end, (instrument, note, measure, beat, note_value)) = label
        audio_onsets.append(start / length * max_frame)
        score_onsets.append(((measure) + float(f'{beat:.6f}')) / n_measures * max_frame)

    # audio_onsets.append(max_frame - 1)
    # score_onsets.append(max_frame - 1)

    return audio_onsets, score_onsets


def extract_annotated_pitches(rec_id, sr=22050, hop_length=512):
    audio, labels = dataset[rec_id]
    hop_length = int(hop_length * (44100 / sr))
    pitches = np.zeros((audio.shape[0] // hop_length, 12 * 11))

    for frame in range(pitches.shape[0]):
        labels_in_frame = labels[frame * hop_length]
        for label in labels_in_frame:
            (start, end, (instrument, note, measure, beat, note_value)) = label
            # print(note)
            pitches[frame, note] += 1

    pitches = np.log(pitches.T + 1)[24:24 + 12 * 8, :]

    return pitches


def extract_audio_pitches(rec_id, sr=22050, hop_length=512):
    pass


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


# def get_alignment(audio_chroma, score_chroma, dist_metric='euclidean'):
#     if dist_metric == 'euclidean':
#         distances = cdist(
#             score_chroma.T,
#             audio_chroma.T
#         )
#     elif dist_metric == 'ta':
#         score_chroma_circular = util.chroma_to_circular(score_chroma)
#         distances = 1 - np.max(np.matmul(score_chroma_circular.T, audio_chroma), axis=1)
#     else:
#         raise ValueError("'{}' is not a valid distance metric!".format(dist_metric))
#     d, wp = dtw.dtw(distances)
#
#     #path = (wp[:, 1], wp[:, 0])
#
#     return distances, wp


def load_rec(rec_id, hop_length=2 ** 14):
    if not os.path.exists(os.path.join(package_dir, 'cache', f'{rec_id}_{str(hop_length)}.npy')):
        print('No cached values found. Building cache:')

        print('Extracting audio...')
        # Extract audio
        audio, labels = dataset[rec_id]
        audio = np.array(audio * 2 ** 15, dtype=np.int16)
        wavfile.write(f'database/musicnet/original/{rec_id}.wav', 44100, audio)
        normal_audio, sr = librosa.load(os.path.join(package_dir, 'original', f'{rec_id}.wav'))
        pitched_audio, sr = librosa.load(os.path.join(package_dir, 'pitched', f'{rec_id}.wav'))
        normal_audio_harm = librosa.effects.harmonic(normal_audio)
        pitched_audio_harm = librosa.effects.harmonic(pitched_audio)
        print('Calculating chroma features for audio...')
        normal_std_chroma = librosa.feature.chroma_stft(normal_audio_harm, sr=sr, hop_length=hop_length)
        normal_gta_chroma = util.pitch_to_chroma(gta.pitch(normal_audio_harm, sr=sr, hop_length=hop_length))
        pitched_std_chroma = librosa.feature.chroma_stft(pitched_audio_harm, sr=sr, hop_length=hop_length)
        pitched_gta_chroma = util.pitch_to_chroma(gta.pitch(pitched_audio_harm, sr=sr, hop_length=hop_length))

        print('Calculating chroma features for score... ')
        score_chroma = util.pitch_to_chroma(
            extract_score_pitches(rec_id, frames=normal_std_chroma.shape[1]))
        ground_truth_path = extract_ground_truth(rec_id, max_frame=normal_std_chroma.shape[1])

        print('Calculating DTWs...')
        print('(1/4)...')
        distances_unpitched_std, path_unpitched_std = get_alignment(normal_std_chroma, score_chroma, 'euclidean')
        # print('(2/4)...')
        # distances_unpitched_gta, path_unpitched_gta = get_alignment(normal_gta_chroma, score_chroma, 'gta')
        # print('(3/4)...')
        # distances_pitched_std, path_pitched_std = get_alignment(pitched_std_chroma, score_chroma, 'euclidean')
        # print('(4/4)...')
        # distances_pitched_gta, path_pitched_gta = get_alignment(pitched_gta_chroma, score_chroma, 'gta')

        print('Saving calculations to cache...')
        cache = (
            normal_std_chroma, normal_gta_chroma,
            pitched_std_chroma, pitched_gta_chroma,
            score_chroma,
            ground_truth_path,
            distances_unpitched_std, path_unpitched_std,
            # distances_unpitched_gta, path_unpitched_gta,
            # distances_pitched_std, path_pitched_std,
            # distances_pitched_gta, path_pitched_gta
        )
        np.save(os.path.join(package_dir, 'cache', f'{rec_id}_{str(hop_length)}.npy'), cache)

        return cache
    else:
        print('Cached values are getting loaded...')
        return np.load(os.path.join(package_dir, 'cache', f'{rec_id}_{str(hop_length)}.npy'))