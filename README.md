# DeepLearningMusic
A beginner at the learning stage of basic concepts and algorithms in the field of Machine Learning, Deep Learning, Artificial Intelligence. 
------------------------------------------------------------------------------------------------------------------------------
This is a very basic neural network code to distinguish between instrumental music and voice music (songs). The code is almost identical to an image classification code ( http://neuralnetworksanddeeplearning.com/chap1.html ) . I just changed the input dataset and modified the code in order to suit the problem in hand. The original code used a 10-neuron output layer to classify the handwritten digits. I have modified that bit to a simple 1-neuron output layer for binary classification to distinguish between songs and instrumental music. 
Instrumental music: 0≤output<0.5
Songs: 0.5≤output≤1
I had to invest time more in generating the input dataset. I am not sharing the dataset here because of the copyright issue. I have used music from my own personal repository, which I have gathered from different CDs, youtube videos, and different websites. I created a small dataset containing ‘30-files training data’, ’10-files validation data’, and ’10-files test data’. Each file was 30 seconds in length in .wav format. I used librosa library to convert the .wav files into simple numpy ndarray and labeled the instrumental music as 0 and voice music as 1 as output dataset. The array of 30 seconds .wav file was too large causing me to get MemoryError. I started offsetting the music files by 10 seconds and finally I got the result for 10 seconds .wav files. I changed the input ndarray dtype to float64 as I was getting RuntimeWarning: overflow encountered in sigmoid function. However, I am still getting that warning for the sigmoid function. As of now, I am ignoring the warning as the code seemed to continue to work. But if anybody can solve this issue, kindly let me know.
I am trying to keep the input dataset small as I am interested in few-shot learning and would like to eventually try out different ideas using small sized dataset.
RESULT:
As for the result, well, it is pretty random as expected. I got accuracy from 30% up to 90% by playing around with the hyper-parameters but could figure no specific pattern out. There was no continuous upward rising or downward moving accuracy trend observed. It was following a zigzag pattern (going up and then down and then again up) in a single run for all tried hyper-parameters and runs. I suppose this kind of outcome is very natural considering the very small training dataset and a very basic neural network algorithm.  
I have only used the numpy array of the .wav files as input dataset here. However, I would try out different attributes of librosa library to extract specific features from the music files and experiment using those. The goal is to ultimately find a method to create orchestral arrangement of the variety of songs from all over the world using deep learning. 
(I’ll be finetuning the code as time goes since there are some parts I hardcoded for time constraint.)