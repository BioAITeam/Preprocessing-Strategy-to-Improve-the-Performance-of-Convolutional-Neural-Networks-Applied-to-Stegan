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


#libreria para diseñar los Modelos de deep learning
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



print("_______________________________________________________________________________________________________________________________________")
print("_______________________________________________gbrasnet_estretegia_final_SUNIWARD___________________________________________")
print("_______________________________________________________________________________________________________________________________________")

## Cargar bases de datos

X_train = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/X_train.npy')
y_train = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/y_train.npy')

X_valid = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/X_valid.npy')
y_valid = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/y_valid.npy')

X_test = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/X_test.npy')
y_test = np.load('/home/rtabares/experimentos_jp/0.4_bpp/SUNIWARD/y_test.npy')

print("datos de entrenamiento: ", X_train.shape)
print("etiquetas de entrenamiento: ", y_train.shape)
print("datos de validacion: ", X_valid.shape)
print("etiquetas de validacion: ", y_valid.shape)
print("datos de test: ", X_test.shape)
print("etiquetas de test: ", y_test.shape)

model = gbrasnet_estretegia_final_paper_DCT_Recortados()
model.summary()


# carpetas para guardar modelos

path_model = "/home/rtabares/experimentos_jp/FINAL/"
model_Name = "gbrasnet_SUNIWARD_04_estretegia_final_paper_DCT_Recortados"

#plot_model(model, show_shapes=True, to_file='multichannel.png')

### Train

train_f(model, X_train, y_train, X_valid, y_valid, X_test, y_test, 32, 500, path_model, model_Name)
