from processing.Mels_preprocess import MelSpectrogramFixed
from .modules.wav2vec2 import Wav2vec2
from .synthesizer import SynthesizerTrn
from hierspeech.speechsr48k.speechsr import SynthesizerTrn as SpeechSR48
from .denoiser.generator import MPNet
from .ttv.ttw2v import TTV_Pitch_Predictor
import os
import torch
import numpy as np
import utils as utils
import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT

# Checkpoint paths
checkpoint_path = 'hierspeech/logs/hierspeechpp_v1.1/hierspeechpp_v1.1_ckpt.pth'
checkpoint_sr24_path = 'hierspeech/speechsr24k/G_340000.pth'
checkpoint_sr48_path = 'hierspeech/speechsr48k/G_100000.pth'
checkpoint_denoise_path = 'hierspeech/denoiser/g_best'
checkpoint_ttv_path = 'hierspeech/logs/ttv/ttv_lt960_ckpt.pth'

# load hyper params
hps = utils.get_hparams_from_file(os.path.join(os.path.split(checkpoint_path)[0], 'config.json'))
h_sr = utils.get_hparams_from_file(os.path.join(os.path.split(checkpoint_sr24_path)[0], 'config.json'))
h_sr48 = utils.get_hparams_from_file(os.path.join(os.path.split(checkpoint_sr48_path)[0], 'config.json'))
hps_denoiser = utils.get_hparams_from_file(os.path.join(os.path.split(checkpoint_denoise_path)[0], 'config.json'))
hps_pitch_predictor = utils.get_hparams_from_file(os.path.join(os.path.split(checkpoint_ttv_path)[0], 'config.json'))


def get_yaapt_f0(audio, rate=16000, interp=False):
    
    frame_length = 20.0
    to_pad = int(frame_length / 1000 * rate) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        signal = basic.SignalObj(y_pad, rate)
        pitch = pYAAPT.yaapt(signal, **{'frame_length': frame_length, 'frame_space': 5.0, 'nccf_thresh1': 0.25,
                                        'tda_frame_length': 25.0, 'f0_max': 1100})
        if interp:
            f0s += [pitch.samp_interp[None, None, :]]
        else:
            f0s += [pitch.samp_values[None, None, :]]
    f0 = np.vstack(f0s)
    return f0


def load_text(fp):
    with open(fp, 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    return filelist


def load_checkpoint(filepath, device):
    print(filepath)
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def add_blank_token(text):
    text_norm = intersperse(text, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def model_load(device):
    mel_fn = MelSpectrogramFixed(
        sample_rate=hps.data.sampling_rate,
        n_fft=hps.data.filter_length,
        win_length=hps.data.win_length,
        hop_length=hps.data.hop_length,
        f_min=hps.data.mel_fmin,
        f_max=hps.data.mel_fmax,
        n_mels=hps.data.n_mel_channels,
        window_fn=torch.hann_window
    ).to(device)
    
    w2v = Wav2vec2().to(device)

    # pitch_predictor = TTV_Pitch_Predictor().to(device)
    # pitch_predictor.load_state_dict(torch.load(checkpoint_ttv_path), strict=False)

    net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                           hps.train.segment_size // hps.data.hop_length,
                           **hps.model).to(device)

    net_g.load_state_dict(torch.load(checkpoint_path, map_location=device))

    _ = net_g.eval()
    speechsr48 = SpeechSR48(h_sr48.data.n_mel_channels,
                            h_sr48.train.segment_size // h_sr48.data.hop_length,
                            **h_sr48.model).to(device)
    utils.load_checkpoint(checkpoint_sr48_path, speechsr48, None, device)
    speechsr48.eval()

    denoiser = MPNet(hps_denoiser).to(device)
    state_dict = load_checkpoint(checkpoint_denoise_path, device)
    denoiser.load_state_dict(state_dict['generator'])
    denoiser.eval()
    return net_g, speechsr48, denoiser, mel_fn, w2v