import logging

import librosa.display

from src import features
from data import *
from src.dtw import *

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def evalute(rec_id, pitched=False, chroma='std', dtw_algo='std', hop_length=1024, test_intervals=None):
    if test_intervals is None:
        test_intervals = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 1]

    logger.debug('Loading audio...')

    if pitched:
        audio, sr = librosa.load(os.path.join(package_dir, 'pitched', f'{rec_id}.wav'))
    else:
        audio = extract_audio(rec_id)

    logger.debug('Computing chroma...')
    if chroma == 'ta':
        audio_chromas = util.pitch_to_chroma(features.audio_to_pitches(audio, hop_length=hop_length))
    elif chroma == 'std':
        audio_chromas = util.pitch_to_chroma(librosa.feature.chroma_cqt(audio, 22050, tuning=0, hop_length=hop_length))

    del audio

    logger.debug('Loading ground truth...')
    length = audio_chromas.shape[1]
    score_chromas = util.pitch_to_chroma(extract_score_pitches(rec_id, length))
    ground_truth = extract_ground_truth(rec_id, length)

    logger.debug('Computing DTW...')
    if dtw_algo == 'std':
        wp = dtw(score_chromas, audio_chromas)
    elif dtw_algo == 'ta':
        wp = ta_dtw(score_chromas, audio_chromas)
    wp_s = np.asarray(wp)

    onsets_audio = np.interp(ground_truth[1], sorted(wp_s[:, 0]), sorted(wp_s[:, 1]))
    diff = np.abs(onsets_audio[10:-10] - ground_truth[0][10:-10])
    diff_s = diff * (hop_length / 22050)

    results = []
    for test in test_intervals:
        count = np.count_nonzero(diff_s <= test)
        percent = count / len(diff_s) * 100
        results.append((count, len(diff_s)))
        print(f'{test:.2f}s:\t{percent:.2f}%', )

    return results


def batch(pitched, algo_chroma, algo_dtw):
    rec_ids = get_ids()
    too_much = np.inf

    for i, rec_id in enumerate(rec_ids):
        is_pitched = 'pitched' if pitched else 'unpitched'

        audio = extract_audio(rec_id)
        length_sec = len(audio) / 44100
        if length_sec >= too_much:
            continue
        if os.path.exists(f'../data/results/{rec_id}_{algo_chroma}_{algo_dtw}_{is_pitched}.npy'):
            continue

        try:
            print(f'================== {rec_id} ==================')
            print(f'Chroma: {algo_chroma}, DTW: {algo_dtw}, Audio: {is_pitched}')
            results = evalute(rec_id, pitched=pitched, dtw_algo=algo_dtw, chroma=algo_chroma)
            np.save(f'../data/results/{rec_id}_{algo_chroma}_{algo_dtw}_{is_pitched}.npy', results)
            print()
            # with open(f'../data/original/{rec_id}_ta.csv', 'w') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(results)
        except MemoryError:
            too_much = length_sec
            print(f'Memory overflow. Length: {int(length_sec)}s. Skipping...')


if __name__ == '__main__':
    batch(True, 'std', 'std')  # Pitched, canonical DTW, standard chroma features
    batch(False, 'std', 'std')  # Unpitched, canonical DTW, standard chroma features
    batch(True, 'ta', 'ta')  # Pitched, TA-DTW, proposed chroma features
    batch(False, 'ta', 'ta')  # Unpitched, TA-DTW, proposed chroma features
    batch(False, 'ta', 'std')  # Unpitched, TA-DTW, standard chroma features
