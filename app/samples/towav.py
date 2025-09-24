import glob 
from pydub import AudioSegment

files = glob.glob('./*/*.mp3')
print(files)

for src in files:
    # Define input and output file paths
    src_mp3 = src
    dst_wav = src.replace('.mp3', '.wav')

    import os
    os.remove(dst_wav)
