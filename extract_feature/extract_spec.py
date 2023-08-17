
import numpy as np
from scipy import io
import librosa
import os

def extract_spec(wavfile, savefile, nfft, hop):
    '''Calculate magnitude spectrogram from raw audio file by librosa.
    
    Args:
        wavfile: The absolute path to the audio file (.wav, .mp4).
        savefile: The absolute path to save the extracted feature in mat format. 
    '''
    y, _sr = librosa.load(wavfile, sr=None)
    M = np.abs(librosa.stft(y, n_fft=nfft, hop_length=hop, window='hamming'))

    data = {'spec': M}
    io.savemat(savefile, data)

    print(savefile, M.shape)

if __name__ == '__main__':
    sr = 16000     # sample rate  16000 (iemocap, daiz_woc) or 44100 (meld, pitt)
    frame = 0.02   # 20ms
    nfft = int(sr*frame)
    hop = nfft//2

    #### use extract_spec
    # wavfile = xxx
    # savefile = xxx

    # extract_spec(wav_file, savefile, nfft, hop)

    state = ['train', 'dev', 'test']

    for s in state:

        wav_root = f'../metadata/wav_data'
        mat_root = f'../metadata/mat_data'

        if not os.path.exists(f'{mat_root}/{s}'):
            os.makedirs(f'{mat_root}/{s}')

        files = os.listdir(f'{wav_root}/{s}')

        for file in files:

            if os.path.exists(f'{mat_root}/{s}/{file}'):
                print(f'file skipped: {file}')
            else:
                extract_spec(f'{wav_root}/{s}/{file}', f'{mat_root}/{s}/{file[:-4]}', nfft, hop)
    
