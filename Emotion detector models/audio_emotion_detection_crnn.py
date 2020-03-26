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


"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.image_conv =nn.Sequential(
        nn.Conv2d(1, 16, 1, padding = (0,0)),
        nn.LeakyReLU(),
        nn.Conv2d(16, 16,1, padding = (0,0)),
        nn.LeakyReLU(),
        )

        self.lstm = nn.LSTM(60*3*16, 1000, 2, batch_first=True)

        self.out_fc = nn.Sequential(
        nn.Linear(1000, 400),
        nn.LeakyReLU(),
        nn.Linear(400,6)
        )

    def forward(self, x ):


        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, 1000).to(device)
        c0 = torch.zeros(2, batch_size, 1000).to(device)

        x = self.image_conv(x.view((-1,1,60,3)))

        #x = x.reshape(-1, sequence_length, 60*3)



        x, _ = self.lstm(x.view((batch_size,-1,60*3*16)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        x = self.out_fc(x[:, -1, :])

        return x
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.image_conv =nn.Sequential(
        nn.Conv2d(2, 32, 3, padding = (1,1)),
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

        x = self.image_conv(x.view((-1,2,60,41)))

        #x = x.reshape(-1, sequence_length, 60*3)



        x, _ = self.lstm(x.view((batch_size,-1,10*15*32)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        #print(x.shape)

        x = self.out_fc(x[:, -1, :]) #gets the last step of the lstm (original shape is batch,step number,hidden_size)

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
    elif "NEU" in name:
        return 3

def get_word_label(name):
    if "IEO" in name:
        return 0
    elif "TIE" in name:
        return 1
    elif "IOM" in name:
        return 2
    elif "IWW" in name:
        return 3
    elif "TAI" in name:
        return 4
    elif "MTI" in name:
        return 5
    elif "IWL" in name:
        return 6
    elif "ITH" in name:
        return 7
    elif "DFA" in name:
        return 8
    elif "ITS" in name:
        return 9
    elif "TSI" in name:
        return 10
    elif "WSI" in name:
        return 11

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
        audio_data+=torch.rand(audio_data.size())

        label = get_subset_label(selected_elem)

        return audio_data,label

#https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-basic_example.html
#https://www.cs.virginia.edu/~vicente/recognition/notebooks/rnn_lab.html
#https://scikit-image.org/docs/dev/api/skimage.io.html

#transform_var=transforms.Compose([Rescale(160),horizontal_flip(160),ToTensor(),illumination_change(),random_noise()])


dataset_train = Audio_dataset(h5_file="data/audio/Ang_hap_sad_delta/audio_features_3lass_delta_train_long.hdf5")

dataset_valid = Audio_dataset(h5_file="data/audio/Ang_hap_sad_delta/audio_features_3class_delta_test_long.hdf5")

def customBatchBuilder(samples):
    audio_data, label = zip(*samples)
    audio_data = pad_sequence(audio_data, batch_first=True, padding_value=0)

    #audio_data = audio_data.view((-1,7,60,41,1))
    label = torch.Tensor(label).int()
    return audio_data,label


batch_size = 64

train_loader = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=3, collate_fn=customBatchBuilder)




valid_loader = DataLoader(dataset_valid, batch_size=batch_size,
                        shuffle=True, num_workers=3, collate_fn=customBatchBuilder)


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

print(inputs.shape)

def train_model(model, criterion, optimizer, num_epochs=25,checkp_epoch=0,scheduler=None):
    since = time.time()

    
    if args.log == "True":
        my_file=open(plot_file, "a")



    pbar=tqdm(range(checkp_epoch,num_epochs))
    for epoch in pbar: #range(checkp_epoch,num_epochs):
        #print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        #print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        #pbar=tqdm(train_loader)
        for sample in train_loader:
            input ,labels = sample
            batch_size = input.size(0)
            input = input.to(device) #.reshape(-1, input.size(0), 60*41)

            labels = labels.long().to(device)



            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):

                outputs = model(input.float())

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                if scheduler!=None:
                    scheduler.step()
        

            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / train_set_size
        train_acc = running_corrects.double() / train_set_size


        model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for sample in valid_loader:
                input ,labels = sample

                input = input.to(device) #.reshape(-1, input.size(0), 60*41).to(device)
                labels = labels.long().to(device)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(False):
                    outputs = model(input.float())
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / valid_set_size
        val_acc = running_corrects.double() / valid_set_size
        """
        for p in model.parameters(): #,model._all_weights[0]): #prints gradients below
            #if n[:6] == 'weight':

            print('===========\ngradient:{}\n----------\n{}----------\n{}'.format(p.grad,p.grad.shape,p.grad.mean()))
        """
        if args.log == "True":
            torch.save({
                 'epoch': epoch,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss
                 },checkpoint_file)


            data = {'epoch': epoch,
            'train_loss': train_loss,
            'train_acc':train_acc.item(),
            'val_loss':val_loss,
            'val_acc':val_acc.item()
            }
            df = pd.DataFrame(data,index=[0])#index=[0] denmezse hata veriyor
            df.to_csv(my_file, header=False,index=False)
        #print()

        pbar.set_description("train acc {:.3} loss {:.4} val acc {:.3} loss {:.4}".format(train_acc,train_loss,val_acc,val_loss))
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
model_ft = Net().to(device)

#model_ft.apply(weights_init)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized

optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-5)

now=datetime.now()

checkpoint_file="emotion_delta3_class"+now.strftime("%d_%m_%Y_%H:%M:%S")+".pt"
plot_file="emotion_plot_delta3_class"+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"


num_epochs=100
"""
checkpoint = torch.load("emotion19_03_2020_01:12:17.pt")


model_ft.load_state_dict(checkpoint['model_state_dict'])
optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])

"""



model_ft = train_model(model_ft, criterion, optimizer_ft,
                      num_epochs=num_epochs,scheduler=None)
