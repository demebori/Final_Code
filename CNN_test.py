from os.path import isdir, join
from pathlib import Path
import pandas as pd
from shutil import copyfile
import gc
# Math
import numpy as np
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
from sklearn.decomposition import PCA
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd

###########################################################

from tensorflow import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from keras.regularizers import l1
import seaborn as sn

batch_size = 64
num_classes = 30

#Neural Network Architecture
proj_model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.4),
  tf.keras.layers.Dense(num_classes)
])

#activity_regularizer=l1(0.001)

# Compile the model
proj_model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)
#proj_model.summary()


#Collect data
training_dataset = tf.keras.utils.image_dataset_from_directory(
    "/content/drive/MyDrive/utveksling/FAG/Malis/Project - Speech recognition/Data/FINISHED_WORD_train",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(int(640*0.1), int(480*0.1)),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    "/content/drive/MyDrive/utveksling/FAG/Malis/Project - Speech recognition/Data/FINISHED_WORD_train",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=batch_size,
    image_size=(int(640*0.1), int(480*0.1)),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

"""""
for image_batch, labels_batch in training_dataset:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
"""
#Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
training_dataset = training_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

epochs = 100

#Train of the model
CNN = proj_model.fit(
  training_dataset,
  verbose=1,
  validation_data=val_dataset,
  epochs = epochs
)


#Visualizing the data
acc = CNN.history['accuracy']
val_acc = CNN.history['val_accuracy']

loss = CNN.history['loss']
val_loss = CNN.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
ax = plt.gca()
ax.set_ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
plt.savefig('img3')




#Collecting data and plotting confusion matrix
predictions = np.array([])
labels =  np.array([])
for x, y in val_dataset:
  predictions = np.concatenate([predictions, np.argmax(proj_model.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

CF = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()

matrix_index = ["no", "cat", "wow", "bed", "right", "happy", "go", "dog", "tree", "nine", "three", "bird", "sheila", "seven", "up", "one", "left", "zero", "stop", "on", "down", "house", "five", "yes", "two", "four", "six", "marvin", "off", "eight" ]

cm_sum = np.sum(CF, axis=1, keepdims=True)
cm_perc = CF / cm_sum.astype(float) * 100
annot = np.empty_like(CF).astype(str)
nrows, ncols = CF.shape
for i in range(nrows):
    for j in range(ncols):
        c = CF[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%' % (p)
        elif c == 0:
            annot[i, j] = ''
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)

#annot[i, j] = '%.1f%%\n%d/%d' % (p c, s)
#Display confusion matrix 
df_cm = pd.DataFrame(CF, index = matrix_index, columns = matrix_index)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig, ax = plt.subplots(figsize=(20,20))
sn.heatmap(df_cm, annot=annot, fmt='')