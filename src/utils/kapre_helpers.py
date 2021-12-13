import librosa
import numpy as np
import tensorflow as tf
from utils.helpers import *
from IPython.display import Audio

from kapre.time_frequency import STFT, InverseSTFT, Magnitude, Phase, MagnitudeToDecibel

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

# This should invert the results of audio_stfts
def reconstruct_audio(chunked_mag, chunked_angle, scaling_factor=-80.):  
    chunked_mag, chunked_angle = chunked_mag.squeeze(3), chunked_angle.squeeze(3)
        
    db, angles = concat_stft(chunked_mag * scaling_factor), concat_stft(chunked_angle)
    db, angles = np.swapaxes(db, 0, 1), np.swapaxes(angles, 0, 1)
    stfts = P2R(librosa.db_to_amplitude(db, ref=200.), angles)
    audio = librosa.istft(stfts)
    return audio

def chunk_stft(stft, chunk_size=88):
    new_stft = np.zeros((stft.shape[0] // chunk_size, chunk_size, stft.shape[1]))
    for i in range(0, stft.shape[0], chunk_size):
        chunk_index  = i // chunk_size
        new_stft[chunk_index, :, :] = stft[i : i + chunk_size, :]
    return new_stft

def concat_stft(chunked_stft, chunk_size=88):
    stft = np.zeros((chunked_stft.shape[0] * chunked_stft.shape[1], chunked_stft.shape[2]))
    
    for i in range(0, chunked_stft.shape[0]):
        stft[i * chunk_size : i*chunk_size + chunk_size,:] = chunked_stft[i, :, :]
    
    return stft

def kapre_stft(x, scaling_factor=-80.):
    x = np.expand_dims(x, 0)
    x = np.expand_dims(x, 2)
    STFT_Layer = STFT(n_fft=2048, win_length=2048, hop_length=512,
                   window_name=None, pad_end=False, pad_begin=True,
                   input_data_format='channels_last', output_data_format='channels_last',
                   input_shape=(len(x), 1))
    ISTFT_Layer = InverseSTFT(n_fft=2048, hop_length=512, 
            input_shape=(88, 1024, 1))
    Mag_Layer = Magnitude()
    Phase_Layer = Phase()
    
    stft = STFT_Layer(x)
    mag = Mag_Layer(stft)

    Decibel_Layer = MagnitudeToDecibel(ref_value=tf.math.reduce_max(mag) ** 2, amin=1e-5 ** 2)
    dec = Decibel_Layer(tf.math.square(mag)) / scaling_factor
    angle = Phase_Layer(stft)
    
    dec = dec.numpy().reshape(dec.shape[1], dec.shape[2])
    angle = angle.numpy().reshape(angle.shape[1], angle.shape[2])
    
    chunk_length = 88
    if dec.shape[0] % 88 != 0:
        multiple = np.ceil(dec.shape[0] / chunk_length)
        pad_amount = chunk_length * multiple - dec.shape[0]
        dec = np.pad(dec, ((0, int(pad_amount)),(0,0)), 'constant', constant_values=(0, 0))
        angle = np.pad(angle, ((0,int(pad_amount)),(0,0)), 'constant', constant_values=(0, 0))
    
    chunked_dec, chunked_angle = chunk_stft(dec), chunk_stft(angle)
    
    chunked_dec, chunked_angle = chunked_dec[:, :, :-1, np.newaxis], chunked_angle[:, :, :-1, np.newaxis]
    return chunked_dec, chunked_angle