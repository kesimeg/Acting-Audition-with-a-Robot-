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

from torch.nn.utils.rnn import pad_sequence

np.random.seed(2)
torch.manual_seed(2)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.image_conv =nn.Sequential(
        nn.Conv2d(1, 64, 1, padding = (0,0)),
        nn.Conv2d(64, 64, 1, padding = (0,0)),
        nn.LeakyReLU(),
        nn.Conv2d(64, 32, 1, padding = (0,0)),
        nn.Conv2d(32, 32, 1, padding = (0,0)),
        nn.LeakyReLU(),
        )

        self.lstm = nn.LSTM(60*3*32, 1000, 2, batch_first=True)

        self.out_fc = nn.Sequential(
        nn.Linear(1000, 400),
        nn.LeakyReLU(),
        nn.Linear(400,4)
        )

    def forward(self, x ):

        
        batch_size = x.size(0)
        h0 = torch.zeros(2, batch_size, 1000).to(device)
        c0 = torch.zeros(2, batch_size, 1000).to(device)

        x = self.image_conv(x.view((-1,1,60,3)))

        #x = x.reshape(-1, sequence_length, 60*3)



        x, _ = self.lstm(x.view((batch_size,-1,60*3*32)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

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


class Audio_video_dataset(Dataset):


    def __init__(self, h5_file, video_root_dir, transform=None):

        self.h5_file = h5_file
        self.data = h5py.File(self.h5_file, 'r')
        self.keys = list(self.data.keys())

        self.video_root_dir = video_root_dir
        self.transform = transform


    def __len__(self):
        return len(self.keys)


    #https://github.com/HHTseng/video-classification/blob/master/CRNN/functions.py

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        selected_elem = self.keys[idx] #current elemen (audio,video)
        sequence_name = os.path.join(self.video_root_dir,
                                selected_elem)

        audio_data = torch.from_numpy(self.data[selected_elem][:])

        video_dir = sequence_name+".flv"

        label = get_label(selected_elem)

        return audio_data,label

#https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/examples/pytorch/pytorch-basic_example.html
#https://www.cs.virginia.edu/~vicente/recognition/notebooks/rnn_lab.html
#https://scikit-image.org/docs/dev/api/skimage.io.html

#transform_var=transforms.Compose([Rescale(160),horizontal_flip(160),ToTensor(),illumination_change(),random_noise()])


dataset_train = Audio_video_dataset(h5_file="../data/audio/audio_features_4class(1pad)_train.hdf5",
                                           video_root_dir='cropped_face_frames',
                                           transform=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor()
                                           ]))

dataset_valid = Audio_video_dataset(h5_file="../data/audio/audio_features_4class(1pad)_test.hdf5",
                                           video_root_dir='cropped_face_frames',
                                           transform=transforms.Compose([
                                              transforms.Resize((64,64)),
                                              transforms.ToTensor()
                                           ]))

def customBatchBuilder(samples):
    audio_data, label = zip(*samples)
    audio_data = pad_sequence(audio_data, batch_first=True, padding_value=0)

    #audio_data = audio_data.view((-1,7,60,41,1))
    label = torch.Tensor(label).int()
    return audio_data,label


batch_size = 64

train_loader = DataLoader(dataset_train, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=customBatchBuilder)




valid_loader = DataLoader(dataset_valid, batch_size=batch_size,
                        shuffle=True, num_workers=0, collate_fn=customBatchBuilder)


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



def train_model(model, criterion, optimizer, num_epochs=25,checkp_epoch=0,scheduler=None):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
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

            input = input.to(device) #.reshape(-1, input.size(0), 60*41)

            labels = labels.long().to(device)



            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):

                outputs = model(input)

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                if scheduler!=None:
                    scheduler.step()
            # statistics

            running_loss += loss.item() * input.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / train_set_size
        train_acc = running_corrects.double() / train_set_size

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #    phase, epoch_loss, epoch_acc))
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
                    outputs = model(input)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(1)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / valid_set_size
        val_acc = running_corrects.double() / valid_set_size
        """
        for p in model.parameters(): #,model._all_weights[0]): #prints gradients below
            #if n[:6] == 'weight':

            print('===========\ngradient:{}\n----------\n{}----------\n{}'.format(p.grad,p.grad.shape,p.grad.mean()))
        """
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

optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)

now=datetime.now()

checkpoint_file="emotion"+now.strftime("%d_%m_%Y_%H:%M:%S")+".pt"
plot_file="emotion_plot"+now.strftime("%d_%m_%Y_%H:%M:%S")+".csv"


num_epochs=100

model_ft = train_model(model_ft, criterion, optimizer_ft,
                      num_epochs=num_epochs,scheduler=None)
