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
import cv2
import queue
from torchvision import datasets, models, transforms,utils
from PIL import Image


send_messagge = False
receive_messagge = False
message_rec = None
message_send = None
record_audio = False
record_finish = False
video_record_finished = False
start_video = False

q = queue.Queue()

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

        self.aud_conv =nn.Sequential(
        nn.Conv2d(1, 32, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        )


        self.img_conv =nn.Sequential(
        nn.Conv2d(1, 32, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, padding = (1,1)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        )
        self.lstm_img = nn.LSTM(16*16*32, 1000, 1, batch_first=True)
        self.lstm_aud = nn.LSTM(10*15*32, 1000, 1, batch_first=True)

        self.out_fc = nn.Sequential(
        nn.Dropout(0.8),
        nn.Linear(2000, 3),
        )

    def forward(self, img,aud ):


        batch_size = 1
        h00 = torch.zeros(1, batch_size, 1000)
        c00 = torch.zeros(1, batch_size, 1000)
        h10 = torch.zeros(1, batch_size, 1000)
        c10 = torch.zeros(1, batch_size, 1000)

        x = self.aud_conv(aud.view((-1,1,60,41)))


        y = self.img_conv(img.view((-1,1,64,64)))



        x, _ = self.lstm_aud(x.view((1,-1,10*15*32)), (h00, c00))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        y, _ = self.lstm_img(y.view((1,-1,16*16*32)), (h10, c10))  # out: tensor of shape (batch_size, seq_length, hidden_size)


        x = self.out_fc(torch.cat((x[:, -1, :],y[:, -1, :]),dim=1)) #gets the last step of the lstm (original shape is batch,step number,hidden_size)



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
    ws.send('{"event_name":"furhatos.event.actions.ActionSpeech","text":"furhat"}')
    ws.send('{"event_name":"furhatos.event.actions.ActionRealTimeAPISubscribe","name":"start_record"}')
    ws.send('{"event_name":"furhatos.event.actions.ActionSpeech","text":"kuzu kuzu"}')
    ws.send('{"event_name":"furhatos.event.actions.ActionRealTimeAPISubscribe","name":"Nod"}')

def thread1(threadname):
    CHUNK = 2**5
    RATE = 22050 #44100
    LEN = 8
    FORMAT = pyaudio.paInt16
    global record_audio

    global video_record_finished
    global start_video
    global img_data
    p = pyaudio.PyAudio()
    model = Net()

    checkpoint_file = "video_emotion3_10fps.pt"
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    emotion_dict = {0:"angry",1:"happy",2:"sad"}

    count =0
    soft = torch.nn.Softmax(dim=1)
    while True:


        if record_audio == True:
            start_video = True
            start = time.time()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)


            print("thread 1 start record",record_audio)
            frames = np.zeros(int(LEN*RATE))
            for i in range(int(LEN*RATE/CHUNK)):

                data = np.frombuffer(stream.read(CHUNK),dtype=np.int16)

                frames[i*32:(i+1)*32]=data/32767
            count+=1
            start_video = False


            stream.stop_stream()
            stream.close()
            #p.terminate()
            print("Audio stream finished",time.time()-start)
            #frames = np.array(frames)
            audio_data = extract_features(frames.astype(float)).unsqueeze(0)
            print("Audio record finished",time.time()-start)
            video_wait = time.time()

            while(video_record_finished==False):
                None

            print("video_wait",time.time()-video_wait)

            model_finish = time.time()

            img_data = q.get()
            print("img shape",img_data.shape)
            outputs = model(img_data,audio_data)
            _, preds = torch.max(outputs, 1)

            print(soft(outputs))
            print(preds.item())
            print(soft(outputs)[0][preds])
            print("thread 1 finish record",record_audio, time.time()-start,preds)
            print("model finish",time.time()-model_finish)
            print()
            ws.send('{"event_name":"emotion_receive","emotion":"%s"}' % emotion_dict[preds.item()])
            video_record_finished =False
            record_audio = False

def thread3(threadname):
    face_cascade = cv2.CascadeClassifier('faces.xml')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(3,320)
    cap.set(4,240)
    transform=transforms.Compose([
       transforms.Resize((64,64)),
       transforms.ToTensor()
    ])
    global start_video
    global video_record_finished
    global img_data
    X = []
    while(True):

        if(record_audio):

            start = time.time()
            while(start_video):
                ret, frame = cap.read()


                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces)!=0:
                    for (x,y,w,h) in faces:
                        imgCrop = gray[y-30:y+h+30,x-30:x+w+30]

                    im = Image.fromarray(np.uint8(imgCrop))
                    image = transform(im)
                    X.append(image)
            if(len(X)!=0 and start_video==False):
                print("X_stack")
                X = torch.stack(X, dim=0)
                print("X shape",X.shape)

                #img_data = X
                q.put(X)
                video_record_finished = True
                X = []
                print("Record time",time.time()-start)
reak

def thread2(threadname):
    global send_messagge
    global receive_messagge
    global message_rec
    global message_send
    global record_audio
    global record_finish
    global audio_class
    while True:
        time.sleep(0.5)

        if receive_messagge == True:
            print("mess received")
            if "start_record" in message_rec:
                record_audio = True
                receive_messagge = False
            print("message false")

thread1 = Thread( target=thread1, args=("Thread-1", ) )
thread2 = Thread( target=thread2, args=("Thread-2", ) )
thread3 = Thread( target=thread3, args=("Thread-3", ) )

websocket.enableTrace(True)
uri = "ws://localhost:8080/websocket"


ws = websocket.WebSocketApp(uri, on_message = on_message, on_close = on_close , on_open = on_open)
wst = Thread(target=ws.run_forever)
wst.daemon = True




thread1.start()
thread2.start()
thread3.start()
wst.start()



thread1.join()
thread2.join()
thread3.join()
wst.join()
