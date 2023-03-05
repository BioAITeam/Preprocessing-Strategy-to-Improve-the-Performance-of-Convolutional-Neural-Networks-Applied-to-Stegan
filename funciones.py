#libraries

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

#librery of the visualization
import matplotlib.pyplot as plt


# Libraries Tensorflow
from tensorflow.keras.layers import Concatenate, Lambda
from tensorflow.keras import backend
from tensorflow.keras.layers import Lambda, Layer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers 
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, ReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, BatchNormalization
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

# libraries for metrics
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


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# time
import datetime

import copy

## _____________________________________function Standarization____________________________________________________________

def samplewise_preprocessing(images,labels):
    filtered_labels=[]
    processed_images = []
    means = []
    stds = []
    for i in range(images.shape[0]):
        mean = np.mean(images[i]) 
        std = np.std(images[i]) 
        if std!=0 and mean != 0:
            means.append(mean)
            stds.append(std)
            processed_images.append((images[i]-mean)/std)
            filtered_labels.append(labels[i])
    
    return np.array(processed_images),np.array(filtered_labels), np.mean(means), np.mean(stds)

def featurewise_preprocessing(images, mean, std):
    processed_images = np.zeros_like(images, dtype=np.float32)
    for i in range(images.shape[0]):
        processed_images[i] = (images[i]-mean)/std
    return processed_images

## _______________________________________________________________________________________________________________________________________

## _____________________________________function for preprocesing____________________________________________________________

def min_max_preprocessing(images,labels):
    filtered_labels=[]
    processed_images = []
    for i in range(len(images)):
        maxi=np.max(images[i])
        mini=np.min(images[i])
        if maxi-mini != 0:
          processed_images.append((images[i]-mini)/(maxi-mini))
          filtered_labels.append(labels[i])
    return np.array(processed_images),np.array(filtered_labels)
## _______________________________________________________________________________________________________________________________________

## _____________________________________function metric____________________________________________________________
def metrics(Y_true,predictions):
    """
    Function to print the performance metrics.

    Inputs
    Y_true: Ground truth labels
    predictions: Predicted labels

    Outputs
    Accuracy, F1 Score, Recall, Precision, Classification report, Confusion matrix
    """

    print('Accuracy:', accuracy_score(Y_true, predictions))
    print('F1 score:', f1_score(Y_true, predictions,average='weighted'))
    print('Recall:', recall_score(Y_true, predictions,average='weighted'))
    print('Precision:', precision_score(Y_true, predictions, average='weighted'))
    print('\n Clasification report:\n', classification_report(Y_true, predictions))
    print('\n Confusion matrix:\n',confusion_matrix(Y_true, predictions))
## _______________________________________________________________________________________________________________________________________

## _____________________________________metrics____________________________________________________________
def get_true_pos(y, pred, th=0.5):

    """
    Function to calculate the total of true positive (TP) predictions.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 1))



def get_true_neg(y, pred, th=0.5):

    """
    Function to calculate the total of true negative (TN) predictions.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 0))



def get_false_neg(y, pred, th=0.5):

    """
    Function to calculate the total of false negative (FN) predictions.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """
    
    pred_t = (pred > th)
    return np.sum((pred_t == False) & (y == 1))



def get_false_pos(y, pred, th=0.5):

    """
    Function to calculate the total of false positive (FP) predictions.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    pred_t = (pred > th)
    return np.sum((pred_t == True) & (y == 0))



def print_confidence_intervals(class_labels, statistics):

    """
    Function to calculate the confidence interval (5%-95%).
    
    Inputs
    class_labels: List with class names
    statistics: 

    Outputs
    Returns DataFrame with confidence intervals for each class
    """

    df = pd.DataFrame(columns=["Mean AUC (CI 5%-95%)"])
    for i in range(len(class_labels)):
        mean = statistics.mean(axis=1)[i]
        max_ = np.quantile(statistics, .95, axis=1)[i]
        min_ = np.quantile(statistics, .05, axis=1)[i]
        df.loc[class_labels[i]] = ["%.2f (%.2f-%.2f)" % (mean, min_, max_)]
    return df



def get_roc_curve(gt, pred, target_names):

    """
    Function to plot the ROC curve.
    
    Inputs
    gt: Ground truth labels
    pred: Predicted labels
    target_names: List with class names 
    """

    for i in range(len(target_names)):
        auc_roc = roc_auc_score(gt[:, i], pred[:, i])
        label = target_names[i] + " AUC: %.3f " % auc_roc
        xlabel = "False positive rate"
        ylabel = "True positive rate"
        a, b, _ = roc_curve(gt[:, i], pred[:, i])
        plt.figure(1, figsize=(7, 7))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(a, b, label=label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper center', bbox_to_anchor=(1.3, 1), fancybox=True, ncol=1)

def plot_calibration_curve(y, pred,class_labels):

    """
    Function to plot the calibration curve.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    class_labels: List with class names 
    """

    plt.figure(figsize=(20, 20))
    for i in range(len(class_labels)):
        plt.subplot(4, 4, i + 1)
        fraction_of_positives, mean_predicted_value = calibration_curve(y[:,i], pred[:,i], n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(mean_predicted_value, fraction_of_positives, marker='.')
        plt.xlabel("Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(class_labels[i])
    plt.tight_layout()
    plt.show()



def get_accuracy(y, pred, th=0.5):

    """
    Function to calculate the accuracy.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    accuracy = 0.0
    TP = get_true_pos(y, pred, th)
    FP = get_false_pos(y, pred, th)
    TN = get_true_neg(y, pred, th)
    FN = get_false_pos(y, pred, th)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    return accuracy



def get_sensitivity(y, pred, th=0.5):

    """
    Function to calculate the sensitivity.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """

    sensitivity = 0.0
    TP = get_true_pos(y,pred,th)
    FN = get_false_neg(y,pred,th)
    sensitivity = TP/(TP+FN)
    return sensitivity


def get_specificity(y, pred, th=0.5):

    """
    Function to calculate the specificity.
    
    Inputs
    y: Ground truth labels
    pred: Predicted labels
    th: Classification threshold
    """
    
    specificity = 0.0
    TN = get_true_neg(y,pred,th)
    FP = get_false_pos(y,pred,th)
    specificity = TN/(TN+FP)
    return specificity



def recall_m(y_true, y_pred):  # métricas opcionales
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):  # métricas opcionales
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def f1_m(y_true, y_pred):   # métrica utilizada durante el entrenamiento, F1-Score
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

##__________________________________________________________________________________________________________________________________________________________________________
## function Train


def train(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, path_model, model_name): 
    start_time = tm.time()
    log_dir=path_model+"/"+model_name+"_"+str(datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":","-"))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    filepath = log_dir+"/saved-model-{epoch:03d}-{val_accuracy:.4f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=None, mode='max')
    model.reset_states()
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, mode='auto', patience=7, min_lr=1e-7)
                              
    global lossTEST
    global accuracyTEST
    global lossTRAIN
    global accuracyTRAIN
    global lossVALID
    global accuracyVALID
    lossTEST,accuracyTEST   = model.evaluate(X_test, y_test,verbose=None)
    lossTRAIN,accuracyTRAIN = model.evaluate(X_train, y_train,verbose=None)
    lossVALID,accuracyVALID = model.evaluate(X_valid, y_valid,verbose=None)

    global history
    model_Name = model_name
    log_Dir = log_dir
    print("Starting the training...")
    history=model.fit(X_train, y_train, epochs=epochs, 
                      callbacks=[checkpoint, tensorboard], 
                      batch_size=batch_size,validation_data=(X_valid, y_valid),verbose=1)
    
    predictions=model.predict(X_test)
    predictions = np.argmax(predictions, axis=-1)
    Y_validation= np.argmax(y_test, axis=-1)

    print(predictions.shape)
    print(Y_validation.shape)

    metrics(Y_validation, predictions)
     
    TIME = tm.time() - start_time
    print("Time "+model_name+" = %s [seconds]" % TIME)

def train_f(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, path_model, model_name): 
    start_time = tm.time()
    path_carpeta = model_name+"_"+str(datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":","-"))
    log_dir=path_model+"/"+ path_carpeta
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    filepath = log_dir+"/saved-model-{epoch:03d}-{val_accuracy:.4f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=None, mode='max')
    model.reset_states()
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, mode='auto', patience=7, min_lr=1e-7)
                              
    global lossTEST
    global accuracyTEST
    global lossTRAIN
    global accuracyTRAIN
    global lossVALID
    global accuracyVALID
    lossTEST,accuracyTEST   = model.evaluate(X_test, y_test,verbose=None)
    lossTRAIN,accuracyTRAIN = model.evaluate(X_train, y_train,verbose=None)
    lossVALID,accuracyVALID = model.evaluate(X_valid, y_valid,verbose=None)

    global history
    model_Name = model_name
    log_Dir = log_dir
    print("Starting the training...")
    history=model.fit(X_train, y_train, epochs=epochs, 
                      callbacks=[checkpoint, tensorboard], 
                      batch_size=batch_size,validation_data=(X_valid, y_valid),verbose=0)
    
    predictions=model.predict(X_test, batch_size=batch_size)
    predictions = np.argmax(predictions, axis=-1)
    Y_validation= np.argmax(y_test, axis=-1)

    print(predictions.shape)
    print(Y_validation.shape)

    metrics(Y_validation, predictions)
     
    TIME = tm.time() - start_time
    print("Time "+model_name+" = %s [seconds]" % TIME)
    print("___________________________________________________________________________________________")
    print("___________________________________________________________________________________________")
    print("____________________________________better model for epochs____________________________________")
    print("___________________________________________________________________________________________")
    print("___________________________________________________________________________________________")
    
    path_carpeta = "/home/rtabares/experimentos_jp/FINAL/"+path_carpeta
    global AccTest
    global LossTest
    AccTest = []
    LossTest= [] 
    B_accuracy = 0 #B --> Best
    for filename in sorted(os.listdir(path_carpeta)):
          if filename != ('train') and filename != ('validation'):
                print(filename)
                model.load_weights(path_carpeta +'/'+ filename)
                loss,accuracy = model.evaluate(X_test, y_test, verbose=0, batch_size=batch_size)
                print(f'Loss={loss:.4f} y Accuracy={accuracy:0.4f} '+'\n')
                BandAccTest  = accuracy
                BandLossTest = loss
                AccTest.append(BandAccTest)    
                LossTest.append(BandLossTest)  
            
                if accuracy > B_accuracy:
                    B_accuracy = accuracy
                    B_loss = loss
                    B_name = filename
    
    print("\n\nBest")
    print(B_name)
    print(f'Loss={B_loss:.4f} y Accuracy={B_accuracy:0.4f}'+'\n')
    print("___________________________________________________________________________________________")
    print("___________________________________________________________________________________________")
       
def train2(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, path_model, model_name): 
    start_time = tm.time()
    log_dir=path_model+"/"+model_name+"_"+str(datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":","-"))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    filepath = log_dir+"/saved-model-{epoch:03d}-{val_accuracy:.4f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=None, mode='max')
    model.reset_states()
    
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, mode='auto', patience=5, min_lr=1e-7)
                              
    global lossTEST
    global accuracyTEST
    global lossTRAIN
    global accuracyTRAIN
    global lossVALID
    global accuracyVALID
    lossTEST,accuracyTEST   = model.evaluate(X_test, y_test,verbose=None)
    lossTRAIN,accuracyTRAIN = model.evaluate(X_train, y_train,verbose=None)
    lossVALID,accuracyVALID = model.evaluate(X_valid, y_valid,verbose=None)

    global history
    model_Name = model_name
    log_Dir = log_dir
    print("Starting the training...")
    history=model.fit(X_train, y_train, epochs=epochs, 
                      callbacks=[checkpoint, tensorboard], 
                      batch_size=batch_size,validation_data=(X_valid, y_valid),verbose=0)
    
    predictions=model.predict(X_test)
    predictions = np.argmax(predictions, axis=-1)
    Y_validation= np.argmax(y_test, axis=-1)

    print(predictions.shape)
    print(Y_validation.shape)

    metrics(Y_validation, predictions)
     
    TIME = tm.time() - start_time
    print("Time "+model_name+" = %s [seconds]" % TIME)
    
def train3(model, X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size, epochs, path_model, model_name):
    start_time = tm.time()
    log_dir=path_model+"/"+model_name+"_"+str(datetime.datetime.now().isoformat()[:19].replace("T", "_").replace(":","-"))
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
    filepath = log_dir+"/saved-model-{epoch:03d}-{val_accuracy:.4f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', save_best_only=False, mode='max')
    model.reset_states()
    
    global lossTEST
    global accuracyTEST
    global lossTRAIN
    global accuracyTRAIN
    global lossVALID
    global accuracyVALID
    lossTEST,accuracyTEST   = model.evaluate(X_test, y_test,verbose=None)
    lossTRAIN,accuracyTRAIN = model.evaluate(X_train, y_train,verbose=None)
    lossVALID,accuracyVALID = model.evaluate(X_valid, y_valid,verbose=None)

    global history
    global model_Name
    global log_Dir
    model_Name = model_name
    log_Dir = log_dir
    print("Starting the training...")
    history=model.fit(X_train, y_train, epochs=epochs, 
                      callbacks=[tensorboard,checkpoint], 
                      batch_size=batch_size,validation_data=(X_valid, y_valid),verbose=2)
    
    metrics = model.evaluate(X_test, y_test, verbose=0)
     
    TIME = tm.time() - start_time
    print("Time "+model_name+" = %s [seconds]" % TIME)
    
    print("\n")
    print(log_dir)
    return {k:v for k,v in zip (model.metrics_names, metrics)}

def Final_Results_Test(PATH_trained_models):
    global AccTest
    global LossTest
    AccTest = []
    LossTest= [] 
    B_accuracy = 0 #B --> Best
    for filename in sorted(os.listdir(PATH_trained_models)):
        if filename != ('train') and filename != ('validation'):
            print(filename)
            model = tf.keras.models.load_model(PATH_trained_models+'/'+filename, custom_objects={'Tanh3':Tanh3})
            loss,accuracy = model.evaluate(X_test, y_test,verbose=0)
            print(f'Loss={loss:.4f} y Accuracy={accuracy:0.4f}'+'\n')
            BandAccTest  = accuracy
            BandLossTest = loss
            AccTest.append(BandAccTest)    
            LossTest.append(BandLossTest)  
            
            if accuracy > B_accuracy:
                B_accuracy = accuracy
                B_loss = loss
                B_name = filename
    
    print("\n\nBest")
    print(B_name)
    print(f'Loss={B_loss:.4f} y Accuracy={B_accuracy:0.4f}'+'\n')

def graphics(history, AccTest, LossTest, log_Dir, model_Name, lossTEST, lossTRAIN, lossVALID, accuracyTEST, accuracyTRAIN, accuracyVALID):
    numbers=AccTest
    numbers_sort = sorted(enumerate(numbers), key=itemgetter(1),  reverse=True)
    for i in range(int(len(numbers)*(0.05))): #5% total epochs
        index, value = numbers_sort[i]
        print("Test Accuracy {}, epoch:{}\n".format(value, index+1))
    
    print("")
    
    numbers=history.history['accuracy']
    numbers_sort = sorted(enumerate(numbers), key=itemgetter(1),  reverse=True)
    for i in range(int(len(numbers)*(0.05))): #5% total epochs
        index, value = numbers_sort[i]
        print("Train Accuracy {}, epoch:{}\n".format(value, index+1))
    
    print("")
    
    numbers=history.history['val_accuracy']
    numbers_sort = sorted(enumerate(numbers), key=itemgetter(1),  reverse=True)
    for i in range(int(len(numbers)*(0.05))): #5% total epochs
        index, value = numbers_sort[i]
        print("Validation Accuracy {}, epoch:{}\n".format(value, index+1))

    with plt.style.context('seaborn-white'):
        plt.figure(figsize=(10, 10))
        plt.plot(np.concatenate([np.array([accuracyTRAIN]),np.array(history.history['accuracy'])],axis=0))
        plt.plot(np.concatenate([np.array([accuracyVALID]),np.array(history.history['val_accuracy'])],axis=0))
        plt.plot(np.concatenate([np.array([accuracyTEST]),np.array(AccTest)],axis=0)) #Test
        plt.title('Accuracy Vs Epoch')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        plt.grid('on')
        plt.savefig(path_img_base+'/Accuracy_GBRAS_Net_'+model_Name+'.eps', format='eps')
        plt.savefig(path_img_base+'/Accuracy_GBRAS_Net_'+model_Name+'.svg', format='svg')
        plt.savefig(path_img_base+'/Accuracy_GBRAS_Net_'+model_Name+'.pdf', format='pdf')     
        plt.show()
        
        plt.figure(figsize=(10, 10))
        plt.plot(np.concatenate([np.array([lossTRAIN]),np.array(history.history['loss'])],axis=0))
        plt.plot(np.concatenate([np.array([lossVALID]),np.array(history.history['val_loss'])],axis=0))
        plt.plot(np.concatenate([np.array([lossTEST]),np.array(LossTest)],axis=0)) #Test
        plt.title('Loss Vs Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        plt.grid('on')
        plt.savefig(path_img_base+'/Loss_GBRAS_Net_'+model_Name+'.eps', format='eps')
        plt.savefig(path_img_base+'/Loss_GBRAS_Net_'+model_Name+'.svg', format='svg')
        plt.savefig(path_img_base+'/Loss_GBRAS_Net_'+model_Name+'.pdf', format='pdf') 
        plt.show()

def trainTPU(path_model, epochs, model_Name, model_):
    global model_name
    start_time = tm.time()
    model_name = model_Name
    path_log_base = path_model+'/'+model_Name
    if not os.path.exists(path_log_base):
        os.makedirs(path_log_base)

    with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
         model = model_

    epoch_ = 1
    for epoch in range(epochs):
        epoch=epoch+1
        print("epoch ",epoch)
        model.fit(X_train,y_train,validation_data=(X_valid,y_valid), batch_size=64*2, epochs=epoch_, verbose=1) 
        model.save_weights(path_model+'/'+model_name+'/'+str(epoch).zfill(4)+'.hdf5', overwrite=True) 

    TIME = tm.time() - start_time
    print("Time "+model_name+" = %s [seconds]" % TIME)
  
 
"""## 30 SRM filters for preprocessing and the activation function"""

################################################## 30 SRM FILTERS
srm_weights = np.load("/home/rtabares/experimentos_jp/SRM_Kernels1.npy") 
biasSRM=np.ones(30)
print (srm_weights.shape)
################################################## TLU ACTIVATION FUNCTION
T3 = 3;
def Tanh3(x):
    tanh3 = K.tanh(x)*T3
    return tanh3
##################################################
 
    
 """## DCT filters function"""

def make_DCT_filter():
    with tf.name_scope("DCT"):
        # Initialize DCT filters
        DCT_filter_n = np.zeros([5, 5, 1, 64])
        # Definition of 8x8 mesh grid
        XX, YY = np.meshgrid(range(5), range(5))
        # DCT basis as filters
        C=np.ones(5)
        C[0]=1/np.sqrt(2)
        for v in range(5):
            for u in range(5):
                DCT_filter_n[:, :, 0, u+v*5]=(2*C[v]*C[u]/5)*np.cos((2*YY+1)*v*np.pi/(10))*np.cos((2*XX+1)*u*np.pi/(10))

        DCT_filter=tf.constant(DCT_filter_n.astype(np.float32))

        return DCT_filter

def Fun(x):
    fun = tf.keras.activations.hard_sigmoid(x)*3
    return fun

#  Strategy function 

def squeeze_excitation_layer(input_layer, out_dim, ratio, conv):
  squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
  excitation = tf.keras.layers.Dense(units=out_dim / ratio, activation='relu')(squeeze)
  excitation = tf.keras.layers.Dense(out_dim,activation='sigmoid')(excitation)
  excitation = tf.reshape(excitation, [-1,1,1,out_dim])
  scale = tf.keras.layers.multiply([input_layer, excitation])
  
  if conv:
    shortcut = tf.keras.layers.Conv2D(out_dim,kernel_size=1,strides=1,
                                      padding='same',kernel_initializer='he_normal')(input_layer)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
  else:
    shortcut = input_layer
  out = tf.keras.layers.add([shortcut, scale])
  return out
    
##_________________________________________________________________________________________________________________________________________________________________________________________________________________________
##_________________________________________________________________________________________________________________________________________________________________________________________________________________________
##_________________________________________________________________________________________________________________________________________________________________________________________________________________________
##________________________________________________________________________________________Models____________________________________________________________________________


def gbrasnet_estretegia_final_paper():
    
    img_size=256
    
    DCT_filter = make_DCT_filter()
    print(DCT_filter.shape)

    filters = np.concatenate([DCT_filter,srm_weights],axis=3)
    filters.shape
    
    bias = np.ones(94)

    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(img_size,img_size,1), name="input")
    # Layer 1
    #conv0 =   tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, activation=Tanh3, use_bias=True)(inputs)

    #Block 1
    layers_ty = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers_tn = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=True, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers1 = tf.keras.layers.add([layers_ty, layers_tn])

    
    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  
    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    block_squeeze = squeeze_excitation_layer(layers, out_dim=94, ratio=32.0, conv=False)

    layers_Tr_1 = tf.keras.layers.add([layers1, block_squeeze])

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)


    layers_Tr_2 = tf.keras.layers.add([layers_Tr_1, layers3])

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_2) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers_n = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_n) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
       
#-------------------------------------------------------------Gbras--------------------------------------------------------------------------------------------------
     #Layer 1

    layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers4)
    #Layer 2
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers1)
    layers = tf.keras.layers.SeparableConv2D(30,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 3
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers)
    layers = tf.keras.layers.SeparableConv2D(30,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers) 
    layers2 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    skip1 =   tf.keras.layers.Concatenate()([layers1, layers2]) 
    #Layer 4
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(skip1) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 5
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 6
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 7
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 8
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers3)
    layers = tf.keras.layers.SeparableConv2D(60,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 9
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers)
    layers = tf.keras.layers.SeparableConv2D(60,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers) 
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    skip2 =   tf.keras.layers.Add()([layers3, layers4])
    #Layer 10
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(skip2) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 11
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 12
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 13
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 14
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 15
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 16
    layers = tf.keras.layers.Conv2D(30, (1,1), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 17
    layers = tf.keras.layers.Conv2D(2, (1,1), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 18
    layers = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(layers)
    #Layer 19
    predictions = tf.keras.layers.Softmax(axis=1)(layers)
    #Model generation
    model = tf.keras.Model(inputs = inputs, outputs=predictions)
    #Optimizer
    optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    print ("Model GBRAS-Net Generated")
    #Model compilation
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def Zhu_Net_estretegia_final_paper():
    
    img_size=256
    
    DCT_filter = make_DCT_filter()
    print(DCT_filter.shape)

    filters = np.concatenate([DCT_filter,srm_weights],axis=3)
    filters.shape
    
    bias = np.ones(94)

    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(img_size,img_size,1), name="input")
    # Layer 1
    #conv0 =   tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, activation=Tanh3, use_bias=True)(inputs)

    #Block 1
    layers_ty = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers_tn = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=True, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers1 = tf.keras.layers.add([layers_ty, layers_tn])

    
    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  
    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    block_squeeze = squeeze_excitation_layer(layers, out_dim=94, ratio=32.0, conv=False)

    layers_Tr_1 = tf.keras.layers.add([layers1, block_squeeze])

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)


    layers_Tr_2 = tf.keras.layers.add([layers_Tr_1, layers3])

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_2) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers_n = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_n) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    
#-------------------------------------------------------------Zhu_Net--------------------------------------------------------------------------------------------------
  
    #Layer 2
    layer2 = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers4)
    layer23 = tf.keras.layers.SeparableConv2D(30,(3,3),activation="relu",depth_multiplier=3,padding="same")(layer2) 
    layer23 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layer23)
    
    #Layer 3
    layer23 = tf.keras.layers.SeparableConv2D(30,(3,3),activation="relu",depth_multiplier=3,padding="same")(layer23)
    layer23 = tf.keras.layers.Lambda(tf.keras.backend.abs)(layer23)
    layer23 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layer23)
    
    #Shorcut
    layers= tf.keras.layers.Add()([layer23, layers4])
    
    #Layer 4
    layers = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation="relu", kernel_initializer='glorot_normal', padding='same')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = tf.keras.layers.AveragePooling2D((5,5), strides= (2,2),padding="same")(layers)
   
    #Layer 5
    layers = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), activation="relu", kernel_initializer='glorot_normal',padding="same")(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = tf.keras.layers.AveragePooling2D((5,5), strides= (2,2),padding="same")(layers)
    
    #Layer 6
    layers = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), activation="relu", kernel_initializer='glorot_normal',padding="same")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = tf.keras.layers.AveragePooling2D((5,5), strides= (2,2),padding="same")(layers)
    
    #Layer 7
    layers = tf.keras.layers.Conv2D(128, (5,5), strides=(1,1), activation="relu", kernel_initializer='glorot_normal',padding="same")(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    
    #Layer 8
    layers = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(layers)
    
    
    #Layer 9, FC, Softmax
    layers = tf.keras.layers.Dense(2688,activation="relu")(layers)
    layers = tf.keras.layers.Dropout(0.2)(layers)
    layers = tf.keras.layers.Dense(1024 ,activation="relu")(layers)
    layers = tf.keras.layers.Dropout(0.2)(layers)
    
    #Softmax
    predictions = tf.keras.layers.Dense(2, activation="softmax", name="output_1")(layers)
    
    #Model generation
    model = tf.keras.Model(inputs = inputs, outputs=predictions)
    
    #Optimizer
    optimizer=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9) 
    
    #Compilator
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print ("Zhu-net model generated")
    return model



def Xunet_estretegia_final_paper():
    
    img_size=256
    
    DCT_filter = make_DCT_filter()
    print(DCT_filter.shape)

    filters = np.concatenate([DCT_filter,srm_weights],axis=3)
    filters.shape
    
    bias = np.ones(94)

    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(img_size,img_size,1), name="input")
    # Layer 1
    #conv0 =   tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, activation=Tanh3, use_bias=True)(inputs)

    #Block 1
    layers_ty = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers_tn = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=True, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers1 = tf.keras.layers.add([layers_ty, layers_tn])

    
    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  
    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    block_squeeze = squeeze_excitation_layer(layers, out_dim=94, ratio=32.0, conv=False)

    layers_Tr_1 = tf.keras.layers.add([layers1, block_squeeze])

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)


    layers_Tr_2 = tf.keras.layers.add([layers_Tr_1, layers3])

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_2) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers_n = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_n) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    
#-------------------------------------------------------------Xunet--------------------------------------------------------------------------------------------------
    #Block 1
    
    #Layer 0
    layers = Conv2D(8, (5,5), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers4) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = Concatenate()([layers, layers, layers])
    
    #Block 2
    
    #Layer 1
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = Conv2D(16, (5,5), strides=1,padding="same", kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)  
    layers = AveragePooling2D((5,5), strides= 2, padding='same')(layers)
    
    #Block 3
    
    #Layer 2
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = Conv2D(32, (1,1), strides=1,padding="same", kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = AveragePooling2D((5,5), strides= 2,padding="same")(layers)
    
    #Block 4
    #Layer 3
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = Conv2D(64, (1,1), strides=1,padding="same", kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = AveragePooling2D((5,5), strides=2,padding="same")(layers)
    #Block 5
    #Layer 4
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = Conv2D(128, (1,1), strides=1,padding="same", kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = Concatenate()([layers, layers, layers])
    layers = GlobalAveragePooling2D(data_format="channels_last")(layers)
    
    #Block 6
    #Layer 5, FC, Softmax
  
    #FC
    layers = Dense(128,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = Dense(64,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
   
    #Softmax
    predictions = Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    model =tf.keras.Model(inputs = inputs, outputs=predictions)
    #Compile
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)
    
    if compile:
        model.compile(optimizer= optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print ("Xunet")
    return model



def Yenet_estretegia_final_paper():
    
    img_size=256
    
    DCT_filter = make_DCT_filter()
    print(DCT_filter.shape)

    filters = np.concatenate([DCT_filter,srm_weights],axis=3)
    filters.shape
    
    bias = np.ones(94)

    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(img_size,img_size,1), name="input")
    # Layer 1
    #conv0 =   tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, activation=Tanh3, use_bias=True)(inputs)

    #Block 1
    layers_ty = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers_tn = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=True, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers1 = tf.keras.layers.add([layers_ty, layers_tn])

    
    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  
    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    block_squeeze = squeeze_excitation_layer(layers, out_dim=94, ratio=32.0, conv=False)

    layers_Tr_1 = tf.keras.layers.add([layers1, block_squeeze])

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)


    layers_Tr_2 = tf.keras.layers.add([layers_Tr_1, layers3])

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_2) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers_n = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_n) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    
#-------------------------------------------------------------Ye-net--------------------------------------------------------------------------------------------------

    #Block 2
    
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers4) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = tf.keras.layers.Concatenate()([layers, layers, layers])
    print(layers.shape)
    
    #Block 3
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    print(layers.shape)
    
    #Block 4
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    print(layers.shape)
    
    #Block 5
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = tf.keras.layers.Conv2D(32, (5,5), strides=(1,1), kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    print(layers.shape)
    
    #Block 6
    layers = tf.keras.layers.Concatenate()([layers, layers, layers])
    layers = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(layers)
    layers = tf.keras.layers.Dense(128,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Dense(64,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    predictions = tf.keras.layers.Dense(2,kernel_initializer='glorot_normal', activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    print(predictions.shape)
    
    #Model generation
    model = tf.keras.Model(inputs = inputs, outputs=predictions)
    #Optimizer
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)#lrate
    #Compilator
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print ("Ye-net model 2 generated")
    return model


   
#-------------------------------------------------------------SR-net--------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

def conv_layer(input_tensor, num_filters, kernel_size, strides, padding='same'):
    
    # He initializer
    filter_initializer = tf.keras.initializers.HeNormal()

    # Bias initializer
    bias_initializer = tf.keras.initializers.Constant(value=0.2)

    # L2 regularization for the filters
    filter_regularizer = tf.keras.regularizers.L2(l2=2e-4)
    
    x = layers.Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding=padding,
                  kernel_initializer=filter_initializer,
                  bias_initializer=bias_initializer,
                  kernel_regularizer=filter_regularizer,
                  use_bias=True)(input_tensor)
    
    return x


def layer_T1_f(input_layer, num_filters):
    # Convolutional layer

    
    DCT_filter = make_DCT_filter()
    print(DCT_filter.shape)

    filters = np.concatenate([DCT_filter,srm_weights],axis=3)
    filters.shape
    
    bias = np.ones(94)

    tf.keras.backend.clear_session()
    input_layer = input_layer
    # Layer 1
    #conv0 =   tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, activation=Tanh3, use_bias=True)(inputs)

    #Block 1
    layers_ty = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, padding='same', activation=Tanh3, use_bias=True)(input_layer)
    layers_tn = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=True, padding='same', activation=Tanh3, use_bias=True)(input_layer)
    layers1 = tf.keras.layers.add([layers_ty, layers_tn])

    
    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  
    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    block_squeeze = squeeze_excitation_layer(layers, out_dim=94, ratio=32.0, conv=False)

    layers_Tr_1 = tf.keras.layers.add([layers1, block_squeeze])

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)


    layers_Tr_2 = tf.keras.layers.add([layers_Tr_1, layers3])

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_2) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers_n = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_n) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    x = conv_layer(layers4, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)

    # ReLU activation layer
    x = tf.keras.layers.ReLU()(x)
    
    return x




def layer_T1(input_tensor, num_filters):
    # Convolutional layer

    x = conv_layer(input_tensor, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)

    # ReLU activation layer
    x = tf.keras.layers.ReLU()(x)
    
    return x


def layer_T2(input_tensor, num_filters):
    # Add the layer T1 to the beginning of Layer T2
    x = layer_T1(input_tensor, num_filters)
    
    # Convolutional layer
    x = conv_layer(x, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    
    # Create the residual connection
    x = layers.add([input_tensor, x]) 
    
    return x


def layer_T3(input_tensor, num_filters):
    # MAIN BRANCH
    # Add the layer T1 to the beginning of Layer T2
    x = layer_T1(input_tensor, num_filters)
    
    # Convolutional layer
    x = conv_layer(x, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    
    # Average pooling layer
    x = tf.keras.layers.AveragePooling2D(pool_size=(3, 3), 
                                strides=2,
                                padding='same')(x)
    
    # SECONDARY BRANCH
    # Special convolutional layer. 
    y = conv_layer(input_tensor, 
                   num_filters=num_filters, 
                   kernel_size=(1, 1), 
                   strides=2)
    
    # Batch normalization layer
    y = tf.keras.layers.BatchNormalization(momentum=0.9)(y)
    
    # Create the residual connection
    output = layers.add([x, y]) 
    
    return output


def layer_T4(input_tensor, num_filters):
    # Add the layer T1 to the beginning of Layer T2
    x = layer_T1(input_tensor, num_filters)
    
    # Convolutional layer
    x = conv_layer(x, 
                   num_filters=num_filters, 
                   kernel_size=(3, 3), 
                   strides=1)
    
    # Batch normalization layer
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    
    # Global Average Pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    return x


def fully_connected(input_tensor):
    
    # Dense weight initializer N(0, 0.01)
    dense_initializer = tf.random_normal_initializer(0, 0.01)
    
    # Bias initializer for the fully connected network
    bias_dense_initializer = tf.constant_initializer(0.)
    
    x = tf.keras.layers.Flatten()(input_tensor)
    x = tf.keras.layers.Dense(512, 
                     activation=None,
                     use_bias=False,
                     kernel_initializer=dense_initializer,
                     bias_initializer=bias_dense_initializer)(x)

        
    output = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    return output


def SRnet_estretegia_final_paper():
    # The input layer has the shape (256, 256, 1)
    #input_layer = layers.Input(shape=input_image_size)
    input_image_size = (256, 256, 1)
    # The input layer has the shape (256, 256, 1)
    input_layer = tf.keras.layers.Input(shape=input_image_size)
    
    x = layer_T1_f(input_layer, 64)
    x = layer_T1(x, 16)
    
    x = layer_T2(x, 16)
    x = layer_T2(x, 16)
    x = layer_T2(x, 16)
    x = layer_T2(x, 16)
    x = layer_T2(x, 16)
    
    x = layer_T3(x, 16)
    x = layer_T3(x, 64)
    x = layer_T3(x, 128)
    x = layer_T3(x, 256)
    
    x = layer_T4(x, 512)
    
    output = fully_connected(x)
    
    model = Model(inputs=input_layer, outputs=output, name="SRNet")

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              optimizer=optimizers.Adamax(learning_rate=0.001),
              metrics=['accuracy'])
    
    return model


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def Yedroud_net_estretegia_final_paper():
    
    img_size=256
    
    DCT_filter = make_DCT_filter()
    print(DCT_filter.shape)

    filters = np.concatenate([DCT_filter,srm_weights],axis=3)
    filters.shape
    
    bias = np.ones(94)

    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(img_size,img_size,1), name="input")
    # Layer 1
    #conv0 =   tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, activation=Tanh3, use_bias=True)(inputs)

    #Block 1
    layers_ty = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers_tn = tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=True, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers1 = tf.keras.layers.add([layers_ty, layers_tn])

    
    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  
    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    block_squeeze = squeeze_excitation_layer(layers, out_dim=94, ratio=32.0, conv=False)

    layers_Tr_1 = tf.keras.layers.add([layers1, block_squeeze])

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(94, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)


    layers_Tr_2 = tf.keras.layers.add([layers_Tr_1, layers3])

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_2) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers_n = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_n) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    
#-------------------------------------------------------------yedroud-net--------------------------------------------------------------------------------------------------


    layers = Thtanh(th=3.0, trainable=False)(layers4)

    
    # Block 1
    
    #Layer 0
    layers = Conv2D(30, (5,5), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = Concatenate()([layers, layers, layers])
    
    # Block 2
    
    #Layer 1
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = Conv2D(30, (5,5), strides=1,padding="same", kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)  
    layers = AveragePooling2D((5,5), strides= 2, padding='same')(layers)
    
    # Block 3
    
    #Layer 2
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = Conv2D(32, (3,3), strides=1,padding="same", kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = AveragePooling2D((5,5), strides= 2,padding="same")(layers)
    
    # Block 4
    #Layer 3
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = Conv2D(64, (3,3), strides=1,padding="same", kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = AveragePooling2D((5,5), strides=2,padding="same")(layers)
    # Block 5
    #Layer 4
    layers = tf.keras.layers.SpatialDropout2D(rate=0.1)(layers)
    layers = Conv2D(128, (3,3), strides=1,padding="same", kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.Lambda(tf.keras.backend.abs)(layers)
    layers = BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    layers = Concatenate()([layers, layers, layers])
    layers = GlobalAveragePooling2D(data_format="channels_last")(layers)
    
    # Block 6
    #Layer 5, FC, Softmax
  
    # FC
    layers = Dense(128,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = Dense(64,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = Dense(32,kernel_initializer='glorot_normal',kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
    layers = ReLU(negative_slope=0.1, threshold=0)(layers)
   
    # Softmax
    predictions = Dense(2, activation="softmax", name="output_1",kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers)
   
    model =tf.keras.Model(inputs = inputs, outputs=predictions)
    # Compile
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.95)#lrate
    
    if compile:
        model.compile(optimizer= optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        print ("Yedroud-net model generated")
        
    return model
    
    
    
    
    
    
    
    
    
##_________________________________________________________________________________________________________________________________________________________________________________________________________________________
##_________________________________________________________________________________________________________________________________________________________________________________________________________________________
##_________________________________________________________________________________________________________________________________________________________________________________________________________________________
##________________________________________________________________________________________Estrategy DCT trimmed____________________________________________________________________________


def make_DCT_filter2():
    with tf.name_scope("DCT"):
        # Initialize DCT filters
        DCT_filter_n = np.zeros([5, 5, 1, 64])
        # Definition of 8x8 mesh grid
        XX, YY = np.meshgrid(range(5), range(5))
        # DCT basis as filters
        C=np.ones(5)
        C[0]=1/np.sqrt(2)
        for v in range(5):
            for u in range(5):
                DCT_filter_n[:, :, 0, u+v*5]=(2*C[v]*C[u]/5)*np.cos((2*YY+1)*v*np.pi/(10))*np.cos((2*XX+1)*u*np.pi/(10))

        DCT_filter=tf.constant(DCT_filter_n.astype(np.float32))

        DCT_filter = DCT_filter[:,:,:,1:]
        DCT_filter = DCT_filter[:,:,:,:24]
        
        
        return DCT_filter
        
        
        
def gbrasnet_estretegia_final_paper_DCT_Recortados():
    
    img_size=256
    
    DCT_filter = make_DCT_filter2()
    print(DCT_filter.shape)

    filters = np.concatenate([DCT_filter,srm_weights],axis=3)
    filters.shape
    
    bias = np.ones(54)

    tf.keras.backend.clear_session()
    inputs = tf.keras.Input(shape=(img_size,img_size,1), name="input")
    # Layer 1
    #conv0 =   tf.keras.layers.Conv2D(94, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, activation=Tanh3, use_bias=True)(inputs)

    #Block 1
    layers_ty = tf.keras.layers.Conv2D(54, (5,5), weights=[filters,bias], strides=(1,1), trainable=False, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers_tn = tf.keras.layers.Conv2D(54, (5,5), weights=[filters,bias], strides=(1,1), trainable=True, padding='same', activation=Tanh3, use_bias=True)(inputs)
    layers1 = tf.keras.layers.add([layers_ty, layers_tn])

    
    layers = tf.keras.layers.Conv2D(54, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
  
    layers = tf.keras.layers.Conv2D(54, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(54, (1,1), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    block_squeeze = squeeze_excitation_layer(layers, out_dim=54, ratio=32.0, conv=False)

    layers_Tr_1 = tf.keras.layers.add([layers1, block_squeeze])

    layers = tf.keras.layers.Conv2D(54, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_1) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(54, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)


    layers_Tr_2 = tf.keras.layers.add([layers_Tr_1, layers3])

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_Tr_2) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers_n = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)

    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1),padding="same", kernel_initializer='glorot_normal', kernel_regularizer=tf.keras.regularizers.l2(0.0001),bias_regularizer=tf.keras.regularizers.l2(0.0001))(layers_n) 
    layers = tf.keras.layers.ReLU(negative_slope=0.1, threshold=0)(layers)
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=True, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
       
#-------------------------------------------------------------Gbras--------------------------------------------------------------------------------------------------
     #Layer 1

    layers1 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers4)
    #Layer 2
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers1)
    layers = tf.keras.layers.SeparableConv2D(30,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 3
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers)
    layers = tf.keras.layers.SeparableConv2D(30,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers) 
    layers2 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    skip1 =   tf.keras.layers.Concatenate()([layers1, layers2]) 
    #Layer 4
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(skip1) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 5
    layers = tf.keras.layers.Conv2D(30, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers)
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 6
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 7
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers3 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 8
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers3)
    layers = tf.keras.layers.SeparableConv2D(60,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 9
    layers = tf.keras.layers.DepthwiseConv2D(1)(layers)
    layers = tf.keras.layers.SeparableConv2D(60,(3,3), padding='same', activation="elu",depth_multiplier=3)(layers) 
    layers4 = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    skip2 =   tf.keras.layers.Add()([layers3, layers4])
    #Layer 10
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(skip2) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 11
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 12
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 13
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 14
    layers = tf.keras.layers.Conv2D(60, (3,3), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 15
    layers = tf.keras.layers.AveragePooling2D((2,2), strides= (2,2))(layers)
    #Layer 16
    layers = tf.keras.layers.Conv2D(30, (1,1), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 17
    layers = tf.keras.layers.Conv2D(2, (1,1), strides=(1,1), activation="elu", padding='same', kernel_initializer='glorot_uniform')(layers) 
    layers = tf.keras.layers.BatchNormalization(momentum=0.2, epsilon=0.001, center=True, scale=False, trainable=True, fused=None, renorm=False, renorm_clipping=None, renorm_momentum=0.4, adjustment=None)(layers)
    #Layer 18
    layers = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(layers)
    #Layer 19
    predictions = tf.keras.layers.Softmax(axis=1)(layers)
    #Model generation
    model = tf.keras.Model(inputs = inputs, outputs=predictions)
    #Optimizer
    optimizer=tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    print ("Model GBRAS-Net Generated")
    #Model compilation
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
