import os
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms, datasets
from torch.utils.data import Dataset
#import torchaudio

import pandas as pd
import numpy as np
import librosa

import torchvision
from torchvision.utils import save_image
import torch.nn.functional as F
import argparse
from torch.optim import lr_scheduler

import scipy.io.wavfile as wav
import time
import copy
import math
from pydub import AudioSegment
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#this code is based on https://github.com/sarthak268/Audio_Classification_using_LSTM
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(sound_clip, bands = 60, frames = 41):
    #window_size = 512 * (frames - 1)
    window_size = 22050//15 #30fps
    log_specgrams = []
    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[int(start):int(end)]) == window_size):
            signal = sound_clip[int(start):int(end)]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.core.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)


    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,-1)
    log_specgrams = np.concatenate((log_specgrams, log_specgrams[-1:]), axis=0) #add last two frames again because of frame mistmatch

    return torch.from_numpy(np.array(log_specgrams))

data_dir = "audio_files_ege_extracted"


arr = os.listdir(data_dir)


#This is the list of the faulty audios/videos
error_array = ["1047_IEO_FEA_LO.wav","1047_IEO_SAD_LO.wav","1056_IEO_ANG_HI.wav","1064_IEO_DIS_MD.wav","1076_MTI_SAD_XX.wav"]

"""
error_array.append("1001_TSI_ANG_XX.wav")
error_array.append("1002_IEO_DIS_HI.wav")
error_array.append("1001_TSI_NEU_XX.wav")
"""
for elem in error_array:
    if elem in arr: arr.remove(elem)

pbar=tqdm(arr)


count = 0
with h5py.File("audio_features_6class.hdf5","w") as hdf:
    for i in pbar:
        count+=1
        sound, _ = librosa.load(os.path.join(data_dir, i))
        features = extract_features(sound)
        if not (features.shape[0]<35  or features.shape[0]>160): #if audio is smaller than 1 sec discard it
            name = str(i.split('.')[0])
            hdf.create_dataset(name,data=features)
