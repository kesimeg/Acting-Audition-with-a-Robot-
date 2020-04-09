from threading import Thread
import time
import asyncio
import websockets
import json
import time
import pyaudio
import asyncio
import websocket
import numpy as np
import wave
import torch
import torch.nn as nn
import librosa

send_messagge = False
receive_messagge = False
message_rec = None
message_send = None
record_audio = False
record_finish = False
audio_class = 5


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
    return torch.from_numpy(np.array(log_specgrams))





class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.image_conv =nn.Sequential(
        nn.Conv2d(1, 32, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(10*15*32, 1000, 1, batch_first=True)

        self.out_fc = nn.Sequential(
        nn.Dropout(0.8),
        nn.Linear(1000, 6),
        )

    def forward(self, x ):


        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, 1000)
        c0 = torch.zeros(1, batch_size, 1000)

        x = self.image_conv(x.view((-1,1,60,41)))

        x, _ = self.lstm(x.view((batch_size,-1,10*15*32)), (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        x = self.out_fc(x[:, -1, :])

        return x

def on_message(ws, messg):
    global receive_messagge
    global message_rec
    message_rec = messg
    print(messg)
    receive_messagge = True


def on_close(ws):
    print("### closed ###")

def on_open(ws):
    ws.send('Connected1')
    ws.send('Connected2')
def thread1(threadname):
    CHUNK = 2**5
    RATE = 22050 #44100
    LEN = 10
    FORMAT = pyaudio.paInt16
    global record_audio
    global audio_class
    p = pyaudio.PyAudio()
    model = Net()

    checkpoint_file = "emotion05_04_2020_17:39:01.pt"
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    class_dict = {0:"Guitar",1:"Violin",2:"Hi-hat",3:"Fireworks",4:"Saxophone",5:"Empyt"}

    soft = torch.nn.Softmax(dim=1)
    while True:


        if record_audio == True:
            #start = time.time()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)


            print("thread 1 start record",record_audio)
            start = time.time()
            frames = np.zeros(int(LEN*RATE))
            for i in range(int(LEN*RATE/CHUNK)): #go for a LEN seconds

                data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)

                frames[i*32:(i+1)*32]=data/32767



            stream.stop_stream()
            stream.close()
            print("Time took",time.time()-start)
            print("Stream closed")
            frames = np.array(frames)

            data = extract_features(frames.astype(float)).unsqueeze(0)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            print(soft(outputs))
            audio_class = class_dict[preds.item()]
            print(audio_class)

            ws.send('message')
            #print("thread 1 finish record",record_audio, time.time()-start,preds)
            record_audio = False
            record_finish = True

def thread3(threadname):
    asyncio.get_event_loop().run_until_complete(send_mess())
    asyncio.get_event_loop().run_until_complete(receive_mess())

def thread2(threadname):
    global send_messagge
    global receive_messagge
    global message_rec
    global message_send
    global record_audio
    global record_finish
    global audio_class
    while True:
        time.sleep(1) 
        #receive_messagge = True #https://pypi.org/project/websocket_client/
      
        if receive_messagge == True:
            print("mess received")
            if "Nod" in message_rec:
                record_audio = True
                receive_messagge = False
            print("message false")


thread1 = Thread( target=thread1, args=("Thread-1", ) )
thread2 = Thread( target=thread2, args=("Thread-2", ) )


websocket.enableTrace(True)
uri = "ws://localhost:8080/websocket"

ws = websocket.WebSocketApp(uri, on_message = on_message, on_close = on_close , on_open = on_open)
wst = Thread(target=ws.run_forever)
wst.daemon = True




thread1.start()
thread2.start()
wst.start()
thread1.join()
thread2.join()
wst.join()
