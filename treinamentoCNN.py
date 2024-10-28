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
        curImg = cv2.imread(path+"/"+str(count)+"/"+arq)#percore as imagem e carrega na memoria e associa ao indice
        images.append(curImg) #nessaa variavem temos todas as imagens, a primeira eh o indice 0 onde tem todas as fotos (joia, assim vai ate atingir todas as classes que temos)
        classNo.append(count) # o indice ou ao numero correspondente a essas imagens

    count +=1

#transformando as imgs em um array np - now py
images = np.array(images)
classNo = np.array(classNo)

#para fazer a separacao da img = Separando as Imagens
#separa img para treinamento e para validacao,  havaliacao de treinamento
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2) #estou tirando 20% de todas as imgs para teste , 80% para treinamento
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2) #20% para validacao

#funcoes de pre-processamento das imagens, etapas principal para fazer o treinamento da rede
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img): #deixar a img mais padronizada possivel
    img = cv2.equalizeHist(img)
    return img
def preprocessing(img): #fazer a normalizacao para cada px
    img = grayscale(img)
    img = equalize(img)
    img=img/255
    return  img

## Pré-processar imagens
X_train=np.array(list(map(preprocessing,X_train)))
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))


## Regularizar Arrays, nivel de profundidade
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

## Aumentando Imagens com ImageDataGenerator, aumenta as imagens do treinamento, os mais bem treinados tem milhares e milhoes de treinamento e nao 700 como esta nesse projeto de cada img
#cria uma nova imagem de acordo com as nossas amostras, isso causa maior generalizacao das imagens
dataGen= ImageDataGenerator(width_shift_range=0.1,   # alterar posição width da imagem
                            height_shift_range=0.1,  # alterar posição hight da imagem
                            zoom_range=0.2,  # colocar zoom
                            shear_range=0.1,  # mudar ângulo
                            rotation_range=10)  # rotacionar imagem
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch = next(batches)

y_train = to_categorical(y_train,noOfClasses) #ajustar o array de acordo com o numero de classes para criar o modelo
y_validation = to_categorical(y_validation,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)

## Criar Modelo
def myModel():
    model= Sequential()
    model.add(Conv2D(32,kernel_size=(5,5),input_shape=(imageDimesions[0],imageDimesions[1],1),activation='relu')) #o melhor modulo para trab com img = activation='relu
    model.add(MaxPooling2D(pool_size=(2,2))) #filtro 2x2

    model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))# dobrou os neuroneos 64 para melhorar enchergar as imgs, caracteristicas melhores das imgs melhor que na primeira camada
    model.add(MaxPooling2D(pool_size=(2,2))) #faz a   extracao mais importante da camada anterior
    model.add(Dropout(0.5)) #destroi neuroneos que sao menos importantes no treinamento (Dropout(0.5))

    model.add(Flatten()) #transforma esse array em uniderecional
    model.add(Dense(128,activation='relu')) #camada densa
    model.add(Dense(noOfClasses,activation='softmax')) #chega nesses parametros fazendo testes, tem que ter conhecimento em redes neurais

    # COMPILE MODEL
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy']) # o melhor compilador para img, buscando a melhor acuracya do modelo
    return model #retorna um modelo com todas as etapas

## Treinamento
model = myModel() #chama a funcao
print(model.summary()) #mostra um resumo


#onde realmente treina o modelo, usando os dados de validacao
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batch_size_val),steps_per_epoch=steps_per_epoch_val,epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1)

## Mostrar histórico de treinamento
#cria um grafico de treinamento para mostrar o historico
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])

## Salvar modelo
model.save('modelo.h5') #salva o modelo em h5
print('Modelo Salvo!')

#demora de 20 a 30 min para ser treinado dependendo da maquia


