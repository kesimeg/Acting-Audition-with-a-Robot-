# Acting-Audition-with-a-Robot-
This is a project for COMP 537 Intelligent User Interfaces class. Project : An acting audition with a robot

In this project a robot is used for an acting audition.

pre_process: Includes codes for train/test split, audio preprocessing for feature extraction

Real time system : Includes a code for real time system. It can send a message and receive a message from robot. Also it can record audio/video and categorize it. All these processes are done in parallel.


Replication:
For this project CREAMA-D dataset has been used. But any other dataset can be used. Only prerocessing code would be needed to change. First download the CREMA-D and create a data  folder. Inside data folder open audio, cropped_faces and raw_audio. Put audio files of CREMA-D to raw_audio. Using preprocess_audio.py extract features. It will create a hd5f file into audio folder.
Then using indep_test_train_valid_split.py split data into three hd5f files for test/train/validation.

Train the model using audio_video_emotion.py. Put the pt file in to Real time system folder.

To interact with the robot run the real_time_system.py before starting the game.

Comp537_Project is the project of the game with the robot. It can be opened with InetelliJ IDEA.
