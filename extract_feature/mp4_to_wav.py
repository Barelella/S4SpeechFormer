import os
import moviepy.editor as mp


inputdir = "D:/Oran/study/master/first year/b/deep learning/project/SpeechFormer/metadata/output_repeated_splits_test"
outputdir = "D:/Oran/study/master/first year/b/deep learning/project/SpeechFormer/metadata/test_wav"

for filename in os.listdir(inputdir):
    actual_filename = filename[:-4]

    if filename.endswith(".mp4"):
        print(f'{inputdir}/{filename}')
        try:
            clip = mp.VideoFileClip(f"{inputdir}/{filename}")
            audio = clip.audio
            audio.write_audiofile(f'{outputdir}/{actual_filename}.wav')
        except:
            print(f'unable to read {filename}')
            continue
    else:
        continue