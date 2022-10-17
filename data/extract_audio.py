import numpy as np
from scipy.io import wavfile

dataset = np.load('musicnet.npz', mmap_mode='r', encoding='latin1', allow_pickle=True)
rec_ids = sorted(dataset.keys())


def extract_audio():
    for i, rec_id in enumerate(rec_ids):
        audio, labels = dataset[rec_id]
        audio = np.array(audio * 2 ** 15, dtype=np.int16)  # convert from float [-1:1] to int16 [-32768:32767
        # labels = sorted(labels)
        # for label in labels:
        #     (start, end, (instrument, note, measure, beat, note_value)) = label

        wavfile.write(rec_id + '.wav', 44100, audio)
        print('{} ({}/{})'.format(rec_id, i + 1, len(rec_ids)))


# extract_audio()
