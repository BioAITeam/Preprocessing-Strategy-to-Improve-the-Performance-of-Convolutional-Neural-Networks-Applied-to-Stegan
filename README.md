# Preprocessing Strategy to Improve the Performance of Convolutional Neural Networks Applied to Steganalysis in the Spatial domain


Recent research has shown that deep learning techniques outperform traditional steganography and steganalysis methods, which has contributed in 
several researches to propose different types of increasingly complex and larger convolutional neural networks (CNNs) to detect steganographic images, 
which aims to outperform the state of the arts most of the time in a 1%-2%. This paper presents a data preprocessing and distribution strategy that improves 
accuracy and convergence during training. The strategy implements a bifurcation of spatial rich model (SRM) filters and DCT filters, which are a set on one 
branch as trainable and on the other untrainable, followed by three blocks of residual convolutions and an excitation layer. The proposed strategy improves 
the accuracy of CNNs applied to steganalysis by 2%-15% while preserving the stability.


## Folders
- **training with TL** this folder contains the codes required to perform the training with transfer learning and multispectral images (15 channels), which contains the python file (code_15channels_with_weights.py) in which the experiments will be run and the file (models_classification2.py) that will contain the models, to be called by the main code.

- **training without TL** this folder contains the codes required to perform the training without transfer learning with multispectral images (15 channels), which contains the python file (code_15channels_without_weights.py ) in which the experiments will be run and the file (models_classification5.py) that will contain the models, to be called by the main code.


## Requirements
This repository requires the following libraries and frameworks:

- TensorFlow 2.10.0
- scikit-learn
- numPy 
- OpenCV 
- Matplotlib
- Time
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
