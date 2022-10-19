import librosa
import numba as nb
import numpy as np


@nb.jit
def pcp(pitch) -> np.ndarray:
    # Reshape sub bands to own axes
    pitch_splitted = np.abs(pitch).reshape(-1, 3, pitch.shape[1])

    energy = pitch_splitted.sum(0)
    energy_argmax = np.argmax(energy, axis=0)

    new_pitch = np.empty((12 * 8, pitch_splitted.shape[2]))

    for frame, feature in enumerate(pitch_splitted.T):
        # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html

        b_index = energy_argmax[frame]
        a_index = (b_index - 1) % 3
        c_index = (b_index + 1) % 3

        a = energy[a_index, frame]
        b = energy[b_index, frame]
        c = energy[c_index, frame]

        if a == b == c == 0.0:
            p = 0
        else:
            p = 0.5 * (a - c) / (a - 2 * b + c)  # Interval: [-0.5, +0.5]

        new_pitch[:, frame] = feature[b_index] - (0.25 * (feature[a_index] - feature[c_index]) * p)

    return new_pitch


def audio_to_pcp(y, hop_length):
    # Calculate pitches with 3 sub bands per semitone
    features = librosa.cqt(y, sr=22050, bins_per_octave=12 * 3, tuning=0, n_bins=12 * 8 * 3, hop_length=hop_length, norm=None)
    features = pcp(features)
    features = pitch_to_chroma(features)

    return features


def audio_to_chroma(y, hop_length):
    features = librosa.feature.chroma_stft(y, sr=22050, tuning=0, hop_length=hop_length, norm=None)
    features = pitch_to_chroma(features)
    return features


def pitch_to_chroma(features):
    chroma = features.reshape(-1, 12, features.shape[1]).sum(0)

    # normalize to length 1
    norms = np.linalg.norm(chroma, axis=0)
    norms[norms == 0] = 1
    chroma /= norms

    return chroma
