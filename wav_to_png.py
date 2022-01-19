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
import files_split
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
import train_set_arrange
import skimage.io
import pylab

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

def wav_to_png(tape_max_len):
    if not os.path.exists("../Data/train2"):
        os.makedirs("../Data/train2")
    else :
        rmtree("../Data/train2")
        os.makedirs("../Data/train2")
    if not os.path.exists("../Data/test2"):
        os.makedirs("../Data/test2")
    else :
        rmtree("../Data/test2")
        os.makedirs("../Data/test2")

    orig_path = "../Data/train/audio"
    path_train = "../Data/train2"
    path_test = "../Data/test2"
    ctr = 0
    ctr_test = 0
    for root, subdirectories, files in os.walk(orig_path):
        random.shuffle(files)

        for file in files :
            path = os.path.join(root, file)
            if file != ".DS_Store":
                m = re.search('/.+/audio[\\\\/](.+?)$', str(root)) #extract word
                m = m.group(1)
                if (not os.path.exists(path_train+"/"+str(m))) and (not os.path.exists(path_test+"/"+str(m))) :
                    os.makedirs(path_train+"/"+str(m))
                    os.makedirs(path_test+"/"+str(m))
                    print(m)
                
                sample_rate, samples = wavfile.read(path)  #Open trace
                
                if(len(samples) < tape_max_len):
                    sample_diff = np.ones(tape_max_len-len(samples))
                    samples=np.concatenate((samples, sample_diff),axis=None) #Pad the amplitude array with zeros

                #Resaple to eliminate unused high frequency range
                new_sample_rate = 8000
                resampled = signal.resample(samples, int(new_sample_rate/sample_rate*samples.shape[0]))

                #Create Mel Power Spectrogram
                S = librosa.feature.melspectrogram(resampled.astype(float), sr=new_sample_rate, n_mels=128)
                #log_S = librosa.power_to_db(S, ref=np.max)
                #Normalize spectrogram values wrt frequency axis
                mean = np.mean(S, axis = 0)
                std = np.std(S, axis = 0)
                S = (S - mean)/std #Normalized logarithmic Mel Power spectrogram

                #img = scale_minmax(S, 0, 255).astype(np.uint8)
                #img = np.flip(img, axis=0) # put low frequencies at the bottom in image
                #img = 255-img # invert. make black==more energy
                if ctr_test < 8 :
                    out = path_train+"/"+str(m)+"/"+str(m)+"_"+str(ctr)+".png"
                else :
                    out = path_test+"/"+str(m)+"/"+str(m)+"_"+str(ctr)+".png"

                ctr_test +=1
                if ctr_test == 10 :
                    ctr_test = 0

                 # save as PNG
                pylab.axis('off')
                pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
                librosa.display.specshow(librosa.power_to_db(S, ref = np.max))
                pylab.savefig(out, bbox_inches = None, pad_inches = 0)
                #skimage.io.imsave(out, S)
                ctr +=1
max_frames = 16000
wav_to_png(max_frames)
