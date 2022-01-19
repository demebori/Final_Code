import os
import sys
from os.path import isdir, join
from pathlib import Path
import pandas as pd
import contextlib
import wave
import collections
import re
from shutil import copyfile, rmtree
import random
# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa

from sklearn.decomposition import PCA

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd
import pylab

#Custom function to convert the .wav files to .png
def wav_to_png(tape_max_len):
    #First thing we create the folder in which we'll create the folders containing the .png files
    if not os.path.exists("../Data/train2"):
        os.makedirs("../Data/train2")
    else :
        rmtree("../Data/train2")
        os.makedirs("../Data/train2")

    orig_path = "../Data/train/audio"
    path_train = "../Data/train2"
    ctr = 0 #counter used for differentiating filenames
    
    #the following loop will walk through all the directories in ../Data/train/audio and 
    #create a corresponding one in ../Data/train2 , in which the newly created .png files will be stored
    for root, subdirectories, files in os.walk(orig_path):
        random.shuffle(files)

        for file in files :
            path = os.path.join(root, file)
            if file != ".DS_Store":
               
                m = re.search('/.+/audio[\\\\/](.+?)$', str(root)) #extract word
                m = m.group(1)
                if (not os.path.exists(path_train+"/"+str(m)))  :
                    os.makedirs(path_train+"/"+str(m)) # create word directory inside ../Data/train2
                    print(m)
                
                sample_rate, samples = wavfile.read(path)  #Open trace
                
                if(len(samples) < tape_max_len):
                    sample_diff = np.ones(tape_max_len-len(samples))
                    samples=np.concatenate((samples, sample_diff),axis=None) #Pad the amplitude array with zeros

                #Resample to eliminate unused high frequency range
                new_sample_rate = 8000
                resampled = signal.resample(samples, int(new_sample_rate/sample_rate*samples.shape[0]))

                #Create Mel Power Spectrogram
                S = librosa.feature.melspectrogram(resampled.astype(float), sr=new_sample_rate, n_mels=128)
                #Normalize spectrogram values with respect to frequency axis
                mean = np.mean(S, axis = 0)
                std = np.std(S, axis = 0)
                S = (S - mean)/std #Normalized logarithmic Mel Power spectrogram

                out = path_train+"/"+str(m)+"/"+str(m)+"_"+str(ctr)+".png"

                 # save as PNG
                pylab.axis('off')
                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
                librosa.display.specshow(librosa.power_to_db(S, ref = np.max))
                pylab.savefig(out, bbox_inches = None, pad_inches = 0)
                #increment image counter
                ctr +=1
                
                
max_frames = 16000
wav_to_png(max_frames)
