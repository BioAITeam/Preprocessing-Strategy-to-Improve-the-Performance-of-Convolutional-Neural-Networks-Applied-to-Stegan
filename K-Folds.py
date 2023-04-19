# -*- coding: utf-8 -*-

#libreria necesarias para cargar los datos y hacer uso de estos.
import numpy as np
from scipy import misc, ndimage, signal
import time
import time as tm
import random 
import ntpath
import os
import pandas as pd
import cv2
import sys
import glob

#libreria para Visualizar datos
import matplotlib.pyplot as plt


#libreria para dise�ar los Modelos de deep learning
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras import backend
from tensorflow.keras.layers import Lambda, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers 
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, SpatialDropout2D, Concatenate, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, BatchNormalization, ReLU
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Libreria para obtener metricas
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import log_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


### binarizar variables categoricas
from sklearn.preprocessing import LabelBinarizer

# libreria para realizar pre procesamientos
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

## libreria para binarizar datos
from sklearn.preprocessing import label_binarize

# libreria para partir los datos en entrenamiento y test
from sklearn.model_selection import train_test_split

# librerias para K-folds
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# graficar modelo creado
from tensorflow.keras.utils import plot_model


# tiempo
import datetime

#segmentar
from scipy import misc
from scipy import ndimage
import copy

#funciones para resumir codigo
from funciones import *

#Kfolds
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


print("_______________________________________________________________________________________________________________________________________")
print("_______________________________________________gbrasnet_estretegia_final_SUNIWARD___________________________________________")
print("_______________________________________________________________________________________________________________________________________")

## Cargar bases de datos

X_train = np.load("/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/X_train.npy")
y_train = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/y_train.npy')

X_valid = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/X_valid.npy')
y_valid = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/y_valid.npy')

X_test = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/X_test.npy')
y_test = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/y_test.npy')


# Crear un objeto LabelBinarizer
binarizador = LabelBinarizer()

# Ajustar y transformar las etiquetas utilizando el binarizador
y_train_binarizadas = binarizador.fit_transform(y_train)
y_valid_binarizadas = binarizador.fit_transform(y_valid)
y_test_binarizadas = binarizador.fit_transform(y_test)


print("datos de entrenamiento: ", X_train.shape)
print("etiquetas de entrenamiento: ", y_train.shape)
print("datos de validacion: ", X_valid.shape)
print("etiquetas de validacion: ", y_valid.shape)
print("datos de test: ", X_test.shape)
print("etiquetas de test: ", y_test.shape)


# carpetas para guardar modelos

path_model = "/home/rtabares/experimentos_jp/FINAL/"
model_Name = "KFolds_gbrasnet_estretegia_final_paper_suniward_04"


# Kfolds


print("________________________________K-folds_________________________________________________")



acc_1 = []
acc_1_sd = []
f1_1 = []
f1_1_sd = []
recall_1 = []
recall_1_sd = []




# Merge inputs and targets
inputs = np.concatenate((X_train, X_valid, X_test), axis=0)
targets = np.concatenate((y_train_binarizadas, y_valid_binarizadas, y_test_binarizadas), axis=0)



print(inputs.shape)
print(targets.shape)

num_folds = 10

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 0

for train, test in kfold.split(inputs, targets):
  fold_no += 1
  # Define the model architecture
  model = Xunet_estretegia_final_paper()

  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {num_folds} ...')

  # Fit data to model
  history = model.fit(inputs[train], targets[train], batch_size=32, epochs=350, verbose=1)

  # Generate generalization metrics
  scores = model.evaluate(inputs[test], targets[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_1.append(scores[1])
  labe1  = np.argmax(targets[test], axis=1) 
  print(labe1.shape)
  pred = model.predict(inputs[test], batch_size=32, verbose=1)  
  predicted = np.argmax(pred, axis=1)
  f1_1.append(f1_score(predicted, labe1, average='macro'))
  recall_1.append(recall_score(predicted, labe1, average='macro'))

  # Increase fold number
  fold_no = fold_no + 1

print("acc", np.mean(acc_1))
print("acc_SD", np.std(acc_1))
print("f1-score", np.mean(f1_1))
print("f1_SD", np.std(f1_1))
print("recall", np.mean(recall_1))
print("recall_SD", np.std(recall_1))

#plot_model(model, show_shapes=True, to_file='multichannel.png')

### Train

#train_f(model, X_train, y_train, X_valid, y_valid, X_test, y_test, 32, 300, path_model, model_Name)

