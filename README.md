# Preprocessing Strategy to Improve the Performance of Convolutional Neural Networks Applied to Steganalysis in the Spatial domain


Recent research has shown that deep learning techniques outperform traditional steganography and steganalysis methods, which has contributed in 
several researches to propose different types of increasingly complex and larger convolutional neural networks (CNNs) to detect steganographic images, 
which aims to outperform the state of the arts most of the time in a 1%-2%. This paper presents a data preprocessing and distribution strategy that improves 
accuracy and convergence during training. The strategy implements a bifurcation of spatial rich model (SRM) filters and DCT filters, which are a set on one 
branch as trainable and on the other untrainable, followed by three blocks of residual convolutions and an excitation layer. The proposed strategy improves 
the accuracy of CNNs applied to steganalysis by 2%-15% while preserving the stability.


## Folders

- **gbrasnet_1.py** This file is an example of training the GbrasNet model designed for the research. In this file, the functions from the "functions.py" file are utilized, followed by the use of the pre-designed functions for training and the model with its implemented strategy.


- **funciones.py** This file will contain the necessary libraries for the proper functioning of the training, strategy, and prediction functions, for the different models used in the research.

- **SRM_Kernels1.npy** This file contains the weights of the 30 SRM high-pass filters used for model training.

- **K-Folds.py** This file contains the method used to perform K-Folds, and obtain a more reliable training result.

## Requirements
This repository requires the following libraries and frameworks:

- TensorFlow 2.10.0
- scikit-learn
- numPy 
- OpenCV 
- Matplotlib
- os
- scikit-image
- glob


This repository was developed in the Python3 (3.9.12) programming language.


## Authors
Universidad Autonoma de Manizales (https://www.autonoma.edu.co/)

- Mario Alejandro Bravo-Ortiz 
- Esteban Mercado-Ruiz 
- Juan Pablo Villa-Pulgarin 
- Harold Brayan Arteaga-Arteaga
- Oscar Cardona
- Gustavo Isaza
- Raúl Ramos-Pollán
- Reinel Tabares-Soto 



## References

[1] 
