import librosa
import numpy as np
from kapre.time_frequency import STFT, InverseSTFT, Magnitude, Phase, MagnitudeToDecibel

STFT_Layer = STFT(n_fft=2048, win_length=2048, hop_length=512,
        window_name=None, pad_end=False, pad_begin=True,
        input_data_format='channels_last', output_data_format='channels_last',
        input_shape=(2048 * 22, 1))
ISTFT_Layer = InverseSTFT(n_fft=2048, hop_length=512, 
        input_shape=(88, 1024, 1))
Mag_Layer = Magnitude()
Phase_layer = Phase()
Decibel_Layer = MagnitudeToDecibel()

# returns magnitude and phase matrices of stft(x)
# in shape of (None, 1024, 88)
def audio_stfts(x):
    x = librosa.util.normalize(x)
    stft = librosa.stft(x)
    mag = librosa.amplitude_to_db(np.abs(stft)) / 80.
    angle = np.angle(stft)
    
    chunk_length = 88
    if mag.shape[1] % 88 != 0:
        multiple = np.ceil(mag.shape[1] / chunk_length)
        pad_amount = chunk_length * multiple - mag.shape[1]
        mag = np.pad(mag, ((0,0),(0, int(pad_amount))), 'constant', constant_values=(0, 0))

        angle = np.pad(angle, ((0,0),(0, int(pad_amount))), 'constant', constant_values=(0, 0))
    
    chunked_mag, chunked_angle = chunk_stft(mag), chunk_stft(angle)
    return chunked_mag[:, :-1, :], chunked_angle[:, :-1, :]

def kapre_stfts(x):
    x = np.expand_dims(x, axis=1)
    stft = STFT_Layer(np.expand_dims(x, axis=0))

    

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

# This should invert the results of audio_stfts
def reconstruct_audio(chunked_mag, chunked_angle):   
    db, angles = concat_stft(chunked_mag), concat_stft(chunked_angle)
    stfts = P2R(librosa.db_to_amplitude(db * 80.), angles)
    audio = librosa.istft(stfts)
    return audio

def chunk_stft(stft, chunk_size=88):
    new_stft = np.zeros((stft.shape[1] // chunk_size, stft.shape[0], chunk_size))
    for i in range(0, stft.shape[1], chunk_size):
        chunk_index  = i // chunk_size
        new_stft[chunk_index, :, :] = stft[:, i : i + chunk_size]
    return new_stft

def concat_stft(chunked_stft, chunk_size=88):
    stft = np.zeros((chunked_stft.shape[1], chunked_stft.shape[0] * chunked_stft.shape[2]))
    
    for i in range(0, chunked_stft.shape[0]):
        stft[:, i * chunk_size : i*chunk_size + chunk_size] = chunked_stft[i, :, :]
    
    return stft    

def return_model_checkpoint(name):
    if name == "reconstruction":
        return ""