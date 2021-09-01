from glob import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import linspace
import soundfile as sf
import time
import IPython
import tensorflow as tf
from perlin_numpy import (
    generate_fractal_noise_2d, generate_perlin_noise_2d, 
)
import json

config_file = f.load('config.json',)
config = json.load(config_file)
training_params = config["training_params"]


learning_rate = 0.0005
num_epochs_to_train =  10
batch_size =  32
vector_dimension = 64

hop=256               #hop size (window size = 4*hop)
sr=44100              #sampling rate
min_level_db=-100     #reference values to normalize data
ref_level_db=20

LEARNING_RATE = learning_rate
BATCH_SIZE = batch_size
EPOCHS = num_epochs_to_train
VECTOR_DIM=vector_dimension

shape=176           #length of time axis of split specrograms         
spec_split=1

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial
import math
import heapq
from torchaudio.transforms import MelScale, Spectrogram

torch.set_default_tensor_type('torch.cuda.FloatTensor')

specobj = Spectrogram(n_fft=4*hop, win_length=4*hop, hop_length=hop, pad=0, power=2, normalized=False)
specfunc = specobj.forward
melobj = MelScale(n_mels=hop, sample_rate=sr, f_min=0.)
melfunc = melobj.forward

def melspecfunc(waveform):
  specgram = specfunc(waveform)
  mel_specgram = melfunc(specgram)
  return mel_specgram

def spectral_convergence(input, target):
    return 20 * ((input - target).norm().log10() - target.norm().log10())

def GRAD(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.002):

    spec = torch.Tensor(spec)
    samples = (spec.shape[-1]*hop)-hop

    if init_x0 is None:
        init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}
    metric_func = spectral_convergence
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.zero_grad()
            V = transform_fn(x)
            loss = criterion(V, T)
            loss.backward()
            optimizer.step()
            lr = lr*0.9999
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    bar_dict[metric] = metric_func(V, spec).item()
                    l2_loss = criterion(V, spec).item()
                    pbar.set_postfix(**bar_dict, loss=l2_loss)
                    pbar.update(evaiter)

    return x.detach().view(-1).cpu()

def normalize(S):
  return np.clip((((S - min_level_db) / -min_level_db)*2.)-1., -1, 1)

def denormalize(S):
  return (((np.clip(S, -1, 1)+1.)/2.) * -min_level_db) + min_level_db

def prep(wv,hop=192):
  S = np.array(torch.squeeze(melspecfunc(torch.Tensor(wv).view(1,-1))).detach().cpu())
  S = librosa.power_to_db(S)-ref_level_db
  return normalize(S)

def deprep(S):
  S = denormalize(S)+ref_level_db
  S = librosa.db_to_power(S)
  wv = GRAD(np.expand_dims(S,0), melspecfunc, maxiter=2500, evaiter=10, tol=1e-8)
  return np.array(np.squeeze(wv))

  #Generate spectrograms from waveform array
def tospec(data):
  specs=[]
  for i in range(data.shape[0]):
    x = data[i]
    S=prep(x)
    S = np.array(S, dtype=np.float32)
    specs.append(np.expand_dims(S, -1))
  return np.array(specs)

#Generate multiple spectrograms with a determined length from single wav file
def tospeclong(path, length=4*sr):
  x, sr = librosa.load(path,sr=sr)
  x,_ = librosa.effects.trim(x)
  loudls = librosa.effects.split(x, top_db=50)
  xls = np.array([])
  for interv in loudls:
    xls = np.concatenate((xls,x[interv[0]:interv[1]]))
  x = xls
  num = x.shape[0]//length
  specs=np.empty(num, dtype=object)
  for i in range(num-1):
    a = x[i*length:(i+1)*length]
    S = prep(a)
    S = np.array(S, dtype=np.float32)
    try:
      sh = S.shape
      specs[i]=S
    except AttributeError:
      print('spectrogram failed')
  print(specs.shape)
  return specs

#Waveform array from path of folder containing wav files
def audio_array(path):
  ls = glob(f'{path}/*.wav')
  adata = []
  for i in range(len(ls)):
    x, sr = tf.audio.decode_wav(tf.io.read_file(ls[i]), 1)
    x = np.array(x, dtype=np.float32)
    adata.append(x)
  return np.array(adata)

#Concatenate spectrograms in array along the time axis
def testass(a):
  but=False
  con = np.array([])
  nim = a.shape[0]
  for i in range(nim):
    im = a[i]
    im = np.squeeze(im)
    if not but:
      con=im
      but=True
    else:
      con = np.concatenate((con,im), axis=1)
  return np.squeeze(con)

#Split spectrograms in chunks with equal size
def splitcut(data):
  ls = []
  mini = 0
  minifinal = spec_split*shape   #max spectrogram length
  for i in range(data.shape[0]-1):
    if data[i].shape[1]<=data[i+1].shape[1]:
      mini = data[i].shape[1]
    else:
      mini = data[i+1].shape[1]
    if mini>=1*shape and mini<minifinal:
      minifinal = mini
  for i in range(data.shape[0]):
    x = data[i]
    if x.shape[1]>=1*shape:
      for n in range(x.shape[1]//minifinal):
        ls.append(x[:,n*minifinal:n*minifinal+minifinal,:])
      ls.append(x[:,-minifinal:,:])
  return np.array(ls)

# Generates timestamp string of "day_month_year_hourMin" 
def get_time_stamp():
  secondsSinceEpoch = time.time()
  timeObj = time.localtime(secondsSinceEpoch)
  x = ('%d_%d_%d_%d%d' % (timeObj.tm_mday, timeObj.tm_mon, timeObj.tm_year, timeObj.tm_hour, timeObj.tm_min))
  return x


def load_raw_data():
    sr = 22050
    k = 10
    X_tr = np.load('gtzan_tr.npy')
    y_tr_dense = X_tr[:,-1]
    y_tr = np.zeros((X_tr.shape[0],k))
    y_tr[range(X_tr.shape[0]),y_tr_dense.astype(int)] = 1
    X_tr = X_tr[:,:-1]
    X_vl = np.load('gtzan_cv.npy')
    y_vl_dense = X_vl[:,-1]
    y_vl = np.zeros((X_vl.shape[0],k))
    y_vl[range(X_vl.shape[0]),y_vl_dense.astype(int)] = 1
    X_vl = X_vl[:,:-1]
    X_ts = np.load('gtzan_te.npy')
    y_ts_dense = X_ts[:,-1]
    y_ts = np.zeros((X_ts.shape[0],k))
    y_ts[range(X_ts.shape[0]),y_ts_dense.astype(int)] = 1
    X_ts = X_ts[:,:-1]
    return X_tr, y_tr, X_vl, y_vl, X_ts, y_ts

def generate_training_data(X_tr):

aspec = tospec(X_tr) 
    new_shape = ((0, 0), (0, 0), (0, 3), (0, 0))
    aspec = np.pad(aspec, pad_width=new_shape, mode='constant', constant_values=0)

    adata = splitcut(aspec)                    #split spectrogams to fixed 

    return adata


def train(x_train, learning_rate, batch_size, epochs, chkpt_pth): 
  vae = VAE(
      input_shape = (hop, shape*spec_split, 1),
      conv_filters=(512, 256, 128, 64, 32),
      conv_kernels=(3, 3, 3, 3, 3),
      conv_strides=(2, 2, 2, 2, (2,1)),
      latent_space_dim = VECTOR_DIM
  )
  vae.summary()
  vae.compile(learning_rate)
  vae.train(x_train, batch_size, epochs, chkpt_pth)
  return vae

def train_tfdata(x_train, learning_rate, epochs=10): 
  vae = VAE(
      input_shape = (hop, 1*shape, 1),
      conv_filters=(512, 256, 128, 64, 32),
      conv_kernels=(3, 3, 3, 3, 3),
      conv_strides=(2, 2, 2, 2, (2,1)),
      latent_space_dim = VECTOR_DIM
  )
  vae.summary()
  vae.compile(learning_rate)
  vae.train(x_train, num_epochs=epochs)
  return vae

def continue_training(checkpoint):
  vae = VAE.load(checkpoint)
  vae.summary()
  vae.compile(LEARNING_RATE)
  vae.train(adata,BATCH_SIZE,EPOCHS)
  return vae

def load_model(checkpoint):
  vae = VAE.load(checkpoint)
  vae.summary()
  vae.compile(LEARNING_RATE)
  return vae


training_run_name = "my_melspecvae_model" #@param {type:"string"}
checkpoint_save_directory = "drive/MyDrive/models/checkpoints/cp.ckpt" #@param {type:"string"}
resume_training = False #@param {type:"boolean"}
resume_training_checkpoint_path = checkpoint_save_directory  #@param {type:"string"}

current_time = get_time_stamp()

if not resume_training:
  vae = train(adata, LEARNING_RATE, BATCH_SIZE, EPOCHS, checkpoint_save_directory)
 #vae = train_tfdata(dsa, LEARNING_RATE, EPOCHS)
  vae.save(f"{checkpoint_save_directory}{training_run_name}_{current_time}_h{hop}_w{shape}_z{VECTOR_DIM}")
else:
  vae = continue_training(resume_training_checkpoint_path)
  vae.save(f"{checkpoint_save_directory}{training_run_name}_{current_time}_h{hop}_w{shape}_z{VECTOR_DIM}")
