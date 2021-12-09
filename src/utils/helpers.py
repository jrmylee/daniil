import librosa
import numpy as np
import pandas as pd
import os
# returns magnitude and phase matrices of stft(x)
# in shape of (None, 1024, 88)
def audio_stfts(x):
    x = librosa.util.normalize(x)
    stft = librosa.stft(x)
    mag = -1 * librosa.amplitude_to_db(np.abs(stft), ref=np.max) / 80.
    angle = np.angle(stft)
    
    chunk_length = 88
    if mag.shape[1] % 88 != 0:
        multiple = np.ceil(mag.shape[1] / chunk_length)
        pad_amount = chunk_length * multiple - mag.shape[1]
        mag = np.pad(mag, ((0,0),(0, int(pad_amount))), 'constant', constant_values=(0, 0))

        angle = np.pad(angle, ((0,0),(0, int(pad_amount))), 'constant', constant_values=(0, 0))
    
    chunked_mag, chunked_angle = chunk_stft(mag), chunk_stft(angle)
    return chunked_mag[:, :-1, :], chunked_angle[:, :-1, :]

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

# This should invert the results of audio_stfts
def reconstruct_audio(chunked_mag, chunked_angle):   
    db, angles = concat_stft(chunked_mag), concat_stft(chunked_angle)
    stfts = P2R(librosa.db_to_amplitude(db * -80.), angles)
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

def load_test_data(ds_path, mapping_filename, num_pieces):    
    csv_path = os.path.join(ds_path, mapping_filename)
    csv = pd.read_csv(csv_path)
    files = []
    
    counter = 0
    for index, row in csv.iterrows():
        if counter == num_pieces:
            break
        full_audio_path = os.path.join(ds_path, row["audio_filename"])

        files.append(full_audio_path)
        
        counter += 1
    
    pieces = [librosa.load(file, sr=44100)[0] for file in files]
    return np.array(pieces)
