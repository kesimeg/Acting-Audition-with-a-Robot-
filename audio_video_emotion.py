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
import torchvision.utils as vutils
from torch.nn.utils.rnn import pad_sequence
import argparse
import random

np.random.seed(2)
torch.manual_seed(2)

parser = argparse.ArgumentParser(description='Emotion detector')

parser.add_argument('--log', type=str, default="True",
                    help='logging of acc. and weights')
args = parser.parse_args()
print(args.log)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.aud_conv =nn.Sequential(
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


        self.img_conv =nn.Sequential(
        nn.Conv2d(1, 32, 3, padding = (1,1)),
        #nn.Conv2d(64, 64, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, padding = (1,1)),
        #nn.Conv2d(32, 32, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )
        self.lstm_img = nn.LSTM(16*16*32, 1000, 1, batch_first=True)
        self.lstm_aud = nn.LSTM(10*15*32, 500, 1, batch_first=True)

        self.out_fc = nn.Sequential(
        nn.Linear(1500, 1000),
        nn.LeakyReLU(),
        nn.Linear(1000,3)
        )

    def forward(self, img,aud ):

        batch_size = 1

        h00 = torch.zeros(1, batch_size, 500).to(device)
        c00 = torch.zeros(1, batch_size, 500).to(device)
        h10 = torch.zeros(1, batch_size, 1000).to(device)
        c10 = torch.zeros(1, batch_size, 1000).to(device)

        x = self.aud_conv(aud.view((-1,1,60,41)))


        y = self.img_conv(img.view((-1,1,64,64)))



        x, _ = self.lstm_aud(x.view((batch_size,-1,10*15*32)), (h00, c00))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        y, _ = self.lstm_img(y.view((batch_size,-1,16*16*32)), (h10, c10))  # out: tensor of shape (batch_size, seq_length, hidden_size)


        x = self.out_fc(torch.cat((x[:, -1, :],y[:, -1, :]),dim=1)) #gets the last step of the lstm (original shape is batch,step number,hidden_size)



        return x


def get_label(name):
    if "ANG" in name:
        return 0
    elif "HAP" in name:
        return 1
    elif "SAD" in name:
        return 2

count = 0

class Audio_video_dataset(Dataset):


    def __init__(self, h5_file, video_root_dir, transform=None):


        self.h5_file = h5_file

        self.audio_dict = {}

        self.frame_list = []

        with  h5py.File(self.h5_file, 'r') as f:
            self.keys = list(f.keys())
            for key in tqdm(self.keys):
                self.audio_dict[key]=f[key][:]
        self.video_root_dir = video_root_dir
        self.transform = transform


        random.shuffle(self.keys)

        video_folder_list = os.listdir(video_root_dir) #[0:1000]
        self.video_list = []

        for i in tqdm(video_folder_list):
            self.video_list = self.video_list  + [i+"/"+x for x in os.listdir(os.path.join(video_root_dir,i))]

    def __len__(self):
        return len(self.keys)

    def read_images(self, video_dir,frame_num):
        image_list=os.listdir(video_dir)

        X = []

        for i in range(1,frame_num+1,3):
            image = Image.open(os.path.join(video_dir,'image-{:d}.jpeg'.format(i)))

            if self.transform:
                image = self.transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        selected_elem = self.keys[idx] #current elemen (audio,video)

        audio_data = torch.from_numpy(self.audio_dict[selected_elem])




        label = get_label(selected_elem)

        sequence_name = os.path.join(self.video_root_dir,selected_elem)

        frame_num = len(os.listdir(sequence_name))

        image_seq = self.read_images(sequence_name,frame_num)

        return image_seq,audio_data,label

#

dataset_train = Audio_video_dataset(h5_file="data/audio/Ang_hap_sad/audio_features_3class_train_long.hdf5",
                                           video_root_dir='data/video/cropped_face_frames',
                                           transform=transforms.Compose([
                                              transforms.Grayscale(),
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor()
                                           ]))
dataset_test = Audio_video_dataset(h5_file="data/audio/Ang_hap_sad/audio_features_3class_test_long.hdf5",
                                           video_root_dir='data/video/cropped_face_frames',
                                           transform=transforms.Compose([
                                           transforms.Grayscale(),
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor()
                                           ]))



batch_size = 5


def customBatchBuilder(samples):
    image_seq,audio_data,label = zip(*samples)
    audio_data = pad_sequence(audio_data, batch_first=True, padding_value=0)
    image_seq = pad_sequence(image_seq, batch_first=True, padding_value=0)

    label = torch.Tensor(label).long()
    return image_seq,audio_data,label



train_loader = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=4,collate_fn=customBatchBuilder)




test_loader = DataLoader(dataset_test, batch_size=batch_size,
                        shuffle=True, num_workers=4,collate_fn=customBatchBuilder)



train_set_size = len(dataset_train)
test_set_size = len(dataset_test)

print("Train set size:",train_set_size)
print("Test set size:",test_set_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # checks if there is gpu available
print(device)


# Get a batch of training data

image_seq,audio_data,label = next(iter(train_loader))
print("SHAPES FROM A RANDOM BATCH..")
print("Label imsize",image_seq.shape)
print("Audio input size",audio_data.shape)


def train_model(model, criterion, optimizer, num_epochs=25,checkp_epoch=0,scheduler=None):
    since = time.time()

    if args.log == "True":
        my_file=open(plot_file, "a")



    pbar=tqdm(range(checkp_epoch,num_epochs))
    for epoch in pbar:
        model.train()

        running_loss = 0.0
        running_corrects = 0

        for sample in train_loader:
            input_img ,input_aud , labels = sample
            batch_size = input_img.size(0)
            input_img = input_img.to(device)
            input_aud = input_aud.to(device)
            labels = labels.long().to(device)


            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(input_img ,input_aud)

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                if scheduler!=None:
                    scheduler.step()
            # statistics

            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / train_set_size
        train_acc = running_corrects.double() / train_set_size

        model.eval()

        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for sample in test_loader:
                input_img ,input_aud ,labels = sample

                input_img = input_img.to(device)
                input_aud = input_aud.to(device)
                labels = labels.long().to(device)

                with torch.set_grad_enabled(False):
                    outputs = model(input_img ,input_aud)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * input_img.size(0)
                running_corrects += torch.sum(preds == labels.data)

        test_loss = running_loss / test_set_size
        test_acc = running_corrects.double() / test_set_size

        if args.log == "True":
            if epoch % 2 ==0:
                torch.save({
                     'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'loss': loss
                     },checkpoint_file+str(epoch)+".pt")


            data = {'epoch': epoch,
            'train_loss': train_loss,
            'train_acc':train_acc.item(),
            'val_loss':test_loss,
            'val_acc':test_acc.item()
            }
            df = pd.DataFrame(data,index=[0])
            df.to_csv(my_file, header=False,index=False)


        pbar.set_description("train acc {:.3} loss {:.4} val acc {:.3} loss {:.4}".format(train_acc,train_loss,test_acc,test_loss))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


    # load best model weights

    return model




model = Net().to(device)


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=1e-4)


now=datetime.now()



checkpoint_file="video_emotion3_10fps"
plot_file="video_audio_emotion"+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"


num_epochs=100
#continue yes add to argparse
if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
        model = train_model(model, criterion, optimizer,
                       num_epochs=num_epochs,checkp_epoch=checkpoint['epoch']+1)
else:
    model = train_model(model, criterion, optimizer,
                       num_epochs=num_epochs)
