from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms,utils
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from datetime  import datetime
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import h5py
import argparse
from torch.nn.utils.rnn import pad_sequence



import librosa



import scipy.io.wavfile as wav


import math
from pydub import AudioSegment

np.random.seed(2)
torch.manual_seed(2)

parser = argparse.ArgumentParser(description='Emotion detection')

parser.add_argument('--path', type=str,
                    help='path of wav file')
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.image_conv =nn.Sequential(
        nn.Conv2d(1, 32, 3, padding = (1,1)),
        #nn.Conv2d(64, 64, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, padding = (1,1)),
        #nn.Conv2d(32, 32, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(10*15*32, 1000, 1, batch_first=True)

        self.out_fc = nn.Sequential(
        nn.Dropout(0.8),
        nn.Linear(1000, 3),
        #nn.LeakyReLU(),
        #nn.Linear(400,6)
        )

    def forward(self, x ):


        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, 1000).to(device)
        c0 = torch.zeros(1, batch_size, 1000).to(device)

        x = self.image_conv(x.view((-1,1,60,41)))

        #x = x.reshape(-1, sequence_length, 60*3)



        x, _ = self.lstm(x.view((batch_size,-1,10*15*32)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        x = self.out_fc(x[:, -1, :])

        return x


def get_label(name): #converts names to int labels
    if "ANG" in name:
        return 0
    elif "HAP" in name:
        return 1
    elif "NEU" in name:
        return 2
    elif "SAD" in name:
        return 3
    elif "DIS" in name:
        return 4
    elif "FEA" in name:
        return 5

def get_subset_label(name):
    if "ANG" in name:
        return 0
    elif "HAP" in name:
        return 1
    elif "SAD" in name:
        return 2
    elif "FEA" in name:
        return 3



#https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-basic_example.html
#https://www.cs.virginia.edu/~vicente/recognition/notebooks/rnn_lab.html
#https://scikit-image.org/docs/dev/api/skimage.io.html

#transform_var=transforms.Compose([Rescale(160),horizontal_flip(160),ToTensor(),illumination_change(),random_noise()])


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)


def extract_features(sound_clip, bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    #window_size = 22050//15 #30fps
    log_specgrams = []
    for (start,end) in windows(sound_clip,window_size):
        if(len(sound_clip[int(start):int(end)]) == window_size):
            signal = sound_clip[int(start):int(end)]
            melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
            logspec = librosa.core.amplitude_to_db(melspec)
            logspec = logspec.T.flatten()[:, np.newaxis].T
            log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,-1)
    #log_specgrams = np.concatenate((log_specgrams, log_specgrams[-1:]), axis=0) #add last two frames again because of frame mistmatch
    #features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    #for i in range(len(features)):
    #    features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    #shape (5,60,41,2)Z
    return torch.from_numpy(np.array(log_specgrams))



#This is the list of the faulty audios/videos




sound, sr = librosa.load(args.path)
inputs = extract_features(sound).unsqueeze(0)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # checks if there is gpu available
print(device)



model = Net().to(device)

#model_ft.apply(weights_init)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized


model_path = "emotion3_class24_03_2020_23:37:43.pt"
checkpoint_file=torch.load(model_path)["model_state_dict"]

model.load_state_dict(checkpoint_file)

model.eval()
print(inputs.shape)
outputs=model(inputs.to(device))
_, preds = torch.max(outputs, 1)
print(preds,get_subset_label(args.path))
print(outputs)
