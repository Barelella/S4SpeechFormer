import os
import numpy as np
from datasets import load_dataset

def extract_huggingface_hubert(layer, wav_file, output_file):


def handle_meld():
    wav_root = '../metadata/wav_data'
    L12_output_root = '../metadata/mat_data/meld/hubert_large_L12_mat'
    L24_output_root = '../metadata/mat_data/meld/hubert_large_L24_mat'

    data_types = ['train',
                  'dev',
                  # 'test'
                  ]
    
    for data_type in data_types:
        wav_dir = f'{wav_root}/{data_type}'
        L12_output_dir = f'{L12_output_root}/{data_type}'#os.path.join(save_L12, s)
        L24_output_dir = f'{L24_output_root}/{data_type}'#os.path.join(save_L24, s)

        if not os.path.exists(L12_output_dir):
            os.makedirs(L12_output_dir)
        if not os.path.exists(L24_output_dir):
            os.makedirs(L24_output_dir)

        samples = os.listdir(wav_dir)
        print(f'We have {len(samples)} samples in total.')
        for sample in samples:
            
            wavfile = f'{wav_dir}/{sample}'
            # [:-4] removes the suffix .wav
            outputfile_L12 = f'{L12_output_dir}/{os.path.splitext(sample)}'#os.path.join(save_L12_s, mat)
            outputfile_L24 = f'{L24_output_dir}/{os.path.splitext(sample)}'#os.path.join(save_L24_s, mat)

            if not os.path.exists(outputfile_L12):
                extract_huggingface_hubert(12, wavfile, outputfile_L12)

            if not os.path.exists(outputfile_L24):
                extract_huggingface_hubert(24, wavfile, outputfile_L24)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    handle_meld()