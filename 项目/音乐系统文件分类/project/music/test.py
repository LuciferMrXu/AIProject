from pydub import AudioSegment
import os
from glob import glob
mp3_p = "./data/music/*.mp3"
music = glob(mp3_p)[0]
print(music)
file = AudioSegment.from_file(music,format="mp3")
file.export(out_f="./data/music/1.wav",format='wav')