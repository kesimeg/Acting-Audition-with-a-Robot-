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

np.random.seed(2)
torch.manual_seed(2)

parser = argparse.ArgumentParser(description='Emotion detection')

parser.add_argument('--log', type=str, default="True",
                    help='logging of acc. and weights')
args = parser.parse_args()
print(args.log)

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

class Audio_dataset(Dataset):


    def __init__(self, h5_file):

        self.h5_file = h5_file

        self.audio_dict = {}
        with  h5py.File(self.h5_file, 'r') as f:
            self.keys = list(f.keys())
            for key in self.keys:
                self.audio_dict[key]=f[key][:]
        #self.keys = list(self.data.keys())
        self.transform = transform


    def __len__(self):
        return len(self.keys)


    #https://github.com/HHTseng/video-classification/blob/master/CRNN/functions.py

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        selected_elem = self.keys[idx] #current elemen (audio,video)

        audio_data = torch.from_numpy(self.audio_dict[selected_elem])

        label = get_subset_label(selected_elem)

        return audio_data,label

#https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-basic_example.html
#https://www.cs.virginia.edu/~vicente/recognition/notebooks/rnn_lab.html
#https://scikit-image.org/docs/dev/api/skimage.io.html

#transform_var=transforms.Compose([Rescale(160),horizontal_flip(160),ToTensor(),illumination_change(),random_noise()])


dataset_train = Audio_dataset(h5_file="data/audio/Ang_hap_sad/audio_features_3class_train_long.hdf5")

dataset_valid = Audio_dataset(h5_file="data/audio/Ang_hap_sad/audio_features_3class_test_long.hdf5")

def customBatchBuilder(samples):
    audio_data, label = zip(*samples)
    audio_data = pad_sequence(audio_data, batch_first=True, padding_value=0)

    #audio_data = audio_data.view((-1,7,60,41,1))
    label = torch.Tensor(label).int()
    return audio_data,label


batch_size = 64

train_loader = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=4, collate_fn=customBatchBuilder)

valid_loader = DataLoader(dataset_valid, batch_size=batch_size,
                        shuffle=True, num_workers=4, collate_fn=customBatchBuilder)


train_set_size = len(dataset_train)

valid_set_size = len(dataset_valid)

print("Train set size:",train_set_size)
print("Valid set size:",valid_set_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # checks if there is gpu available
print(device)

"""
def forward(self, x):

    batch_size, timesteps, C,H, W = x.size()
    c_in = x.view(batch_size * timesteps, C, H, W)

    c_out = self.cnn(c_in)

    r_out, (h_n, h_c) = self.rnn(c_out.view(-1,batch_size,c_out.shape[-1]))

    logits = self.classifier(r_out)

    return logits
"""
#class_names = dataset_train.classes


def imshow(inp, title=None):
    #"Imshow for Tensor.
    #inp = inp.numpy()[0,0,:,:,:]
    #print(inp.shape)
    inp = inp.transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()
    #plt.pause(1)  # pause a bit so that plots are updated



# Get a batch of training data

inputs, labels = next(iter(train_loader))



# Make a grid from batch

#out = torchvision.utils.make_grid(inputs) #belli bir bölümünü ekrana bas (resmin son halini görmek için)

#imshow(inputs)



def test_model(model, criterion):
    since = time.time()


    model.eval()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0
    ####################################################
    class_num = 4
    ####################################################
    confusion_matrix = torch.zeros(class_num, class_num)
    ####################################################
    # Iterate over data.
    pbar=tqdm(train_loader)
    with torch.no_grad():
        for sample in pbar:
            input ,labels = sample

            input = input.to(device)

            labels = labels.long().to(device)

            outputs = model(input)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            #print(labels.data == preds)
            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(preds == labels.data)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    print(confusion_matrix)
    train_loss = running_loss / train_set_size
    train_acc = running_corrects.double() / train_set_size

    running_loss = 0.0
    running_corrects = 0
    confusion_matrix = torch.zeros(class_num, class_num)
    with torch.no_grad():
        pbar=tqdm(valid_loader)
        for sample in pbar:
            input ,labels = sample

            input = input.to(device) #.reshape(-1, input.size(0), 60*41).to(device)
            labels = labels.long().to(device)
            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(input)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix.diag()/confusion_matrix.sum(1))
    print(confusion_matrix)
    val_loss = running_loss / valid_set_size
    val_acc = running_corrects.double() / valid_set_size


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    # load best model weights

    return model


"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

"""
model = Net().to(device)

#model_ft.apply(weights_init)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized


model_path = "emotion3_class24_03_2020_23:37:43.pt"
checkpoint_file=torch.load(model_path)["model_state_dict"]

model.load_state_dict(checkpoint_file)



model = test_model(model, criterion)
