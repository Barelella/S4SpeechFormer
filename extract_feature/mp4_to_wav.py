import os
import moviepy.editor as mp


inputdir = "../MELD.Raw/train/train_splits/"
outputdir = "../MELD.Raw/train/train_splits_wav/"

for filename in os.listdir(inputdir):
    actual_filename = filename[:-4]

    if filename.endswith(".mp4"):
        print(f'{inputdir}/{filename}')
        try:
            clip = mp.VideoFileClip(f"{inputdir}/{filename}")
            audio = clip.audio
            audio.write_audiofile(f'{outputdir}/{actual_filename}.wav')
        except Exception as e:
            print(f'unable to read {filename}, exception: {e}')
            continue
    else:
        continue