from pathlib import Path

import numpy as np
from scipy.io import wavfile

dataset = np.load('musicnet.npz', mmap_mode='r', encoding='latin1', allow_pickle=True)
rec_ids = sorted(dataset.keys())

if __name__ == '__main__':
    output_dir = Path('./original/')
    output_dir.mkdir(exist_ok=True)
    for i, rec_id in enumerate(rec_ids):
        audio, labels = dataset[rec_id]
        audio = np.array(audio * 2 ** 15, dtype=np.int16)  # convert from float [-1:1] to int16 [-32768:32767

        wavfile.write(str(output_dir / f'{rec_id}.wav'), 44100, audio)
        print('{} ({}/{})'.format(rec_id, i + 1, len(rec_ids)))
