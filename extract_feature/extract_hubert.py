import fairseq
# import soundfile as sf
import scipy.signal as signal
from scipy import io
import torch
import torch.nn.functional as F
import os
import numpy as np

## packages for pretrained hubert

def get_receptive_field(k: list, s: list):
    k.reverse()
    s.reverse()

    output_1 = 1
    output_2 = 2
    for _k, _s in zip(k, s):
        recept_1 = (output_1 - 1) * _s + _k
        output_1 = recept_1
        recept_2 = (output_2 - 1) * _s + _k
        output_2 = recept_2

    print('After the convolutional waveform encoder in HuBERT, the feature')
    print('receptive field is:', recept_1, 'points (/sr -> second)')
    print('hop is:', recept_2 - recept_1, 'points (/sr -> second)')

    
class Hubert(object):
    def __init__(self, ckpt_path, max_chunk=1600000, wav_length=104640):
        # fairseq outputs a "model ensemble" - a list of models really
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        # Then they take the first one
        self.model = model[0].eval().cuda()
        self.task = task
        self.max_chunk = max_chunk
        self.wav_length = wav_length  # (326 + 1) * 0.02 * 16000 = 104640

    def read_audio(self, path):
        wav, sr = sf.read(path)
        
        if sr != self.task.cfg.sample_rate:
            num = int((wav.shape[0]) / sr * self.task.cfg.sample_rate)
            wav = signal.resample(wav, num)
            print(f'Resample {sr} to {self.task.cfg.sample_rate}')
        
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim

        return wav
    
    def read_audio_batch(self, path_batch):
        x = []
        for path in path_batch:
            _x = self.read_audio(path)
            _x = np.pad(_x, (0, self.wav_length - _x.shape[0]), constant_values=(0, 0)) if _x.shape[0] < self.wav_length else _x[:self.wav_length]
            x.append(_x)

        x = np.stack(x, axis=0)
        return x

    def get_features(self, path, layer: int):
        '''Layer index starts from 1. (e.g. 1-24)
        '''
        if isinstance(path, str):
            B = 1
            x = self.read_audio(path)
        else:
            B = len(path)
            x = self.read_audio_batch(path)

        x = torch.from_numpy(x).float().cuda()
        if self.task.cfg.normalize:
            x = F.layer_norm(x, x.shape)
        x = x.view(B, -1)

        feat = []
        for start in range(0, x.size(1), self.max_chunk):
            x_chunk = x[:, start: start + self.max_chunk]
            feat_chunk, _ = self.model.extract_features(
                source=x_chunk,
                padding_mask=None,
                mask=False,
                output_layer=layer,
            )
            feat.append(feat_chunk)
        return torch.cat(feat, 1)

def extract_hubert(model: Hubert, layer: int, wavfile: str, savefile: str):
    """ This function exctracts features based on HuBERT model 

    Args:
        model (Hubert): model in class Hubert
        layer (int): layer - choice between 12 and 24
        wavfile (_type_): _description_
        savefile (_type_): _description_
    """
    with torch.no_grad():
        features = model.get_features(wavfile, layer=layer).squeeze(0)

    features = features.cpu().detach().numpy()   # (t, 768)  / (t, 1024)
    dict = {'hubert': features}
    io.savemat(savefile, dict)
    
    print(f'{savefile} -> {features.shape}')

def handle_iemocap(model: Hubert):
    matroot = '/148Dataset/data-chen.weidong/iemocap/feature/wav_wav2vec_mat'
    save_L12 = '/148Dataset/data-chen.weidong/iemocap/feature/hubert_large_L12_mat'
    save_L24 = '/148Dataset/data-chen.weidong/iemocap/feature/hubert_large_L24_mat'

    if not os.path.exists(save_L12):
        os.makedirs(save_L12)
    if not os.path.exists(save_L24):
        os.makedirs(save_L24)

    mats = os.listdir(matroot)
    print(f'We have {len(mats)} samples in total.')
    for mat in mats:
        ses = mat[4]
        folder = mat[:-5]
        wavfile = f'/148Dataset/data-chen.weidong/iemocap/Session{ses}/sentences/wav/{folder}/{mat}.wav'
        savefile_L12 = os.path.join(save_L12, mat)
        savefile_L24 = os.path.join(save_L24, mat)
        extract_hubert(model, 12, wavfile, savefile_L12)
        extract_hubert(model, 24, wavfile, savefile_L24)

def handle_meld(model: Hubert):
    wav_root = '../metadata/wav_data'
    save_L12 = '../metadata/mat_data/meld/hubert_large_L12_mat'
    save_L24 = '../metadata/mat_data/meld/hubert_large_L24_mat'

    state = ['train'
             , 'dev'
             # , 'test'
             ]

    for state in state:
        wav_root_str = f'{wav_root}/{state}'#os.path.join(matroot, s)
        save_L12_s = f'{save_L12}/{state}'#os.path.join(save_L12, s)
        save_L24_s = f'{save_L24}/{state}'#os.path.join(save_L24, s)

        if not os.path.exists(save_L12_s):
            os.makedirs(save_L12_s)
        if not os.path.exists(save_L24_s):
            os.makedirs(save_L24_s)

        samples = os.listdir(wav_root_str)
        print(f'We have {len(samples)} samples in total.')
        for sample in samples:

            wavfile = f'matroot/{state}/{sample}'
            # [:-4] removes the suffix .wav
            outputfile_L12 = f'{save_L12_s}/{sample[:-4]}'#os.path.join(save_L12_s, mat)
            outputfile_L24 = f'{save_L24_s}/{sample[:-4]}'#os.path.join(save_L24_s, mat)

            if os.path.exists(outputfile_L12):
                print(f'file skipped: {sample}')

            else:
                extract_hubert(model, 12, wavfile, outputfile_L12)
                extract_hubert(model, 24, wavfile, outputfile_L24)

def handle_pitt(model: Hubert):
    matroot = '/148Dataset/data-chen.weidong/DementiaBank/Pitt/feature/wav_wav2vec_mat'
    save_L12 = '/148Dataset/data-chen.weidong/DementiaBank/Pitt/feature/hubert_large_L12_mat'
    save_L24 = '/148Dataset/data-chen.weidong/DementiaBank/Pitt/feature/hubert_large_L24_mat'

    state = ['Control', 'Dementia']
    for s in state:
        matroot_s = os.path.join(matroot, s, 'cookie')
        save_L12_s = os.path.join(save_L12, s, 'cookie')
        save_L24_s = os.path.join(save_L24, s, 'cookie')

        if not os.path.exists(save_L12_s):
            os.makedirs(save_L12_s)
        if not os.path.exists(save_L24_s):
            os.makedirs(save_L24_s)

        mats = os.listdir(matroot_s)
        print(f'We have {len(mats)} samples in total.')
        for mat in mats:
            wavfile = f'/148Dataset/data-chen.weidong/DementiaBank/Pitt/audio/utterance_wav/{s}/cookie/{mat}.wav'
            savefile_L12 = os.path.join(save_L12_s, mat)
            savefile_L24 = os.path.join(save_L24_s, mat)
            extract_hubert(model, 12, wavfile, savefile_L12)
            extract_hubert(model, 24, wavfile, savefile_L24)

def handle_daic(model: Hubert):
    matroot = '/148Dataset/data-chen.weidong/AVEC2017/feature/wav_wav2vec_mat'
    save_L12 = '/148Dataset/data-chen.weidong/AVEC2017/feature/hubert_large_L12_mat'
    save_L24 = '/148Dataset/data-chen.weidong/AVEC2017/feature/hubert_large_L24_mat'

    if not os.path.exists(save_L12):
        os.makedirs(save_L12)
    if not os.path.exists(save_L24):
        os.makedirs(save_L24)

    mats = os.listdir(matroot)
    print(f'We have {len(mats)} samples in total.')
    for mat in mats:
        wavfile = f'/148Dataset/data-chen.weidong/AVEC2017/audio/separate_wav/{mat}_AUDIO.wav'
        savefile_L12 = os.path.join(save_L12, mat)
        savefile_L24 = os.path.join(save_L24, mat)
        extract_hubert(model, 12, wavfile, savefile_L12)
        extract_hubert(model, 24, wavfile, savefile_L24)


# def force_cudnn_initialization():
#     s = 32
#     dev = torch.device('cuda')
#     torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


if __name__ == '__main__':
    # force_cudnn_initialization()
    # torch.cuda.empty_cache()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    get_receptive_field(k=[10, 3, 3, 3, 3, 2, 2], s=[5, 2, 2, 2, 2, 2, 2])
    
    ckpt_path = "../pre_trained_model/hubert/hubert_large_ll60k.pt"  # hubert_large_ll60k, hubert_base_ls960
    model = Hubert(ckpt_path)

    # handle_iemocap(model)
    handle_meld(model)
    # handle_pitt(model)
    # handle_daic(model)
    
