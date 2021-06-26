# -- encoding:utf-8 --
"""
Create by ibf on 2018/10/27
"""

import glob
from pydub import AudioSegment

file_paths = glob.glob('./data/*.wav')
for idx, file_path in enumerate(file_paths):
    song = AudioSegment.from_file(file_path)
    song = song.set_channels(channels=2)
    song = song.set_frame_rate(frame_rate=44100)
    song = song.set_sample_width(sample_width=2)
    song = song[:31000]
    song.export('./data/input/{}.wav'.format(idx), format='wav')
