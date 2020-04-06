# Acting-Audition-with-a-Robot-
This is a project for COMP 537 Intelligent User Interfaces class. Project : An acting audition with a robot

In this project a robot is used for an acting audition.

Emotion detector models: Includes training codes for two different architectures.

Pre process codes: Includes codes for train/test split, audio preprocessing for feature extraction

Real time system : Includes codes for real time system. There are two files. One file records audio and uses an lstm to classify the audio. The other code receives and sends messages to robot. These two codes will be eventually get together.

class_accs.py: Calculates class individual accuracies and calculates the confusion matrix.

one_wav_emotion.py : Uses a wav file as input, outputs the coressponding emotion.



