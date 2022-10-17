import os
import math
import random
import shutil

import reapy
from reapy import reascript_api as RPR
from tqdm import tqdm

SHAPE = 2  # Squared
# SHAPE = 0  # Linear
MAX_SEMITONES = 4
TIME_STEP = 5
RENDER_TO = '/home/simon/Documents/REAPER Media/test.wav'


def process_wave(path, output):
    os.makedirs(output, exist_ok=True)
    filename = os.path.basename(path)
    target_path = os.path.join(output, filename)
    if os.path.exists(target_path):
        return

    with reapy.inside_reaper():
        project = reapy.Project()
        track = project.tracks[0]
        if track.n_items > 0:
            track.items[0].delete()

        project.cursor_position = 0  # Place cursor at beginning as it is used as insert position
        RPR.InsertMedia(path, 0)  # Insert current MIDI file to first track
        item = track.items[0]
        take = item.takes[0]

        length_seconds = math.ceil(RPR.GetMediaItemInfo_Value(item.id, "D_LENGTH"))
        num_steps = int(length_seconds / TIME_STEP)
        tqdm.write(f'{num_steps}\t{target_path}')

        RPR.Main_OnCommand(41612, 0)  # Toggle Pitch Envelope
        pitch = take.envelopes[0]

        cur_pitch = random.randint(-2, 2)
        for time in range(-1, length_seconds, TIME_STEP):
            RPR.InsertEnvelopePoint(pitch.id, time, cur_pitch, SHAPE, 0, False, True)
            cur_pitch += random.triangular(-1, 1, 0)
            cur_pitch = min(MAX_SEMITONES, cur_pitch)
            cur_pitch = max(-MAX_SEMITONES, cur_pitch)

        project.cursor_position = 0  # Place cursor at beginning
        RPR.Main_OnCommand(41622, 0)  # Zoom to selected media

        RPR.Main_OnCommand(42230, 0)  # Render with last settings and close dialog afterwards
        item.delete()  # ... and delete it

        shutil.move(RENDER_TO, target_path)


if __name__ == '__main__':
    from glob import glob

    for path in tqdm(glob('/home/simon/Repositories/ta-dtw/data/original/*.wav')):
        process_wave(path, output="/home/simon/Repositories/ta-dtw/data/drift/")
