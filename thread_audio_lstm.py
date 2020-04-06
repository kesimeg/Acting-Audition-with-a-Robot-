from threading import Thread
import json
import time
import pyaudio
import asyncio
import websockets
import numpy as np
import wave
import torch
import torch.nn as nn
import librosa

send_messagge = True
receive_messagge = False
message = None
record_audio = False


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
        nn.Linear(1000, 6),
        #nn.LeakyReLU(),
        #nn.Linear(400,6)
        )

    def forward(self, x ):


        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, 1000)
        c0 = torch.zeros(1, batch_size, 1000)

        x = self.image_conv(x.view((-1,1,60,41)))

        #x = x.reshape(-1, sequence_length, 60*3)



        x, _ = self.lstm(x.view((batch_size,-1,10*15*32)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        x = self.out_fc(x[:, -1, :])

        return x


def thread1(threadname):
    CHUNK = 2**5
    RATE = 22050 #44100
    LEN = 10
    FORMAT = pyaudio.paInt16
    global record_audio
    p = pyaudio.PyAudio()
    model = Net()

    checkpoint_file = "emotion05_04_2020_17:39:01.pt"
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    count =0
    soft = torch.nn.Softmax(dim=1)
    while True:

        global record_audio
        if record_audio == True:
            start = time.time()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
            #frames = []

            print("thread 1 start record",record_audio)
            frames = np.zeros(int(LEN*RATE))
            for i in range(int(LEN*RATE/CHUNK)): #go for a LEN seconds

                data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)
                #frames.append(data/)
                frames[i*32:(i+1)*32]=data/32767
            """
            wf = wave.open("deneme{:d}.wav".format(count), 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            """
            count+=1

            stream.stop_stream()
            stream.close()
            #p.terminate()
            frames = np.array(frames)

            data = extract_features(frames.astype(float)).unsqueeze(0)
            print(data.shape)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            print(soft(outputs))
            print(outputs.shape)
            print("thread 1 finish record",record_audio, time.time()-start,preds)
            record_audio = False






def thread2(threadname):
    global send_messagge
    global receive_messagge
    global message
    global record_audio
    while True:


        time.sleep(20)

        record_audio = True
        #print(1)

        #send_messagge = True
        """
        if receive_messagge == True:
            print("thread",message)
            receive_messagge = False
            global record_audio
            record_audio = True
            print(record_audio)
        """

thread1 = Thread( target=thread1, args=("Thread-1", ) )
thread2 = Thread( target=thread2, args=("Thread-2", ) )

thread1.start()
thread2.start()

thread1.join()
thread2.join()
