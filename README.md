# Acting-Audition-with-a-Robot-
This is a project for COMP 537 Intelligent User Interfaces class. Project : An acting audition with a robot

In this project a robot is used for an acting audition.

pre_process: Includes codes for train/test split, audio preprocessing for feature extraction

Real time system : Includes a code for real time system. It can send a message and receive a message from robot. Also it can record audio/video and categorize it. All these processes are done in parallel.


Replication:
For this project CREAMA-D dataset has been used. But any other dataset can be used. Only prerocessing code would be needed to change. First download the CREMA-D and extract features using preprocess_audio.py. This will create a hd5f file in data. This file will be splitted in to test/train/validation using indep_test_train_valid_split.py
