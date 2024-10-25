import os

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2

#usar essas versoes
#tf versao 2.0.0
#keras versao 2.3.1

#Parametros
path = 'Imagens' #onde estao as amostras
batch_size_val=50 #tamanho do lote durante o treinamento, qtidade de img que serao utilizados durante a interacao
#a rede neural vai extrair as caracteristcas de 50 em 50 img
steps_per_epoch_val=1000 #interacoes para treinarmento
epochs_val=10  #epocas qtidade de vezes que o treinamento sera feito
imageDimesions = (32,32,3) #dimensoes da img de entrada, img com 3 canais RGB

#Importar Imagens
count = 0
images = []
classNo = []
pastas = os.listdir(path) #lista as imgs na pasta
print("Total de Classes:",len(pastas)) #3 pastas , 3 classes sao as pastas das imgs
noOfClasses=len(pastas) #para identidicar as pastas automaticamente

for pt in range(0,len(pastas)):
    arquivos = os.listdir(path+"/"+str(count))
    for arq in arquivos:
        curImg = cv2.imread(path+"/"+str(count)+"/"+arq)
        images.append(curImg)
        classNo.append(count)

    count +=1


