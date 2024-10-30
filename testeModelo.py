
#teste do modelo em tempo real

import numpy as np
import cv2
from keras.models import load_model #vai fazer a importacao do modelo que foi ssalvo .h5

cap = cv2.VideoCapture(0) # webcan

model = load_model('modelo.h5')  # Carrega o modelo sem problemas de compatibilidade


#aplica o mesmo pre processamento de treinamento, tem que ser o mesmo
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

#titulo nas imagens, para cada indice de imagem
def getCalssName(classNo):
    if classNo == 0:
        return 'JOIA'
    elif classNo == 1:
        return 'PAZ E AMOR'
    elif classNo == 2:
        return 'ROCK AND ROLL'

while True:
    success, imgOrignal = cap.read()

    img = np.asarray(imgOrignal) #treinamento
    img = cv2.resize(img, (32, 32)) #tamanho da imagem
    img = preprocessing(img) #pre processamento
    img = img.reshape(1, 32, 32, 1) #redireciona com a camada no array


    predictions = model.predict(img)  #modelo predicti, fazer a predicao da img da webcan
    indexVal = np.argmax(predictions) #vai pegar a img com maior probabilidade
    probabilityValue = np.amax(predictions) #valor de probabilidade
    print(indexVal,probabilityValue)

    #classe
    cv2.putText(imgOrignal, str(getCalssName(indexVal)), (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 8, cv2.LINE_AA)
    #probabilidade
    cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (120, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2,
                cv2.LINE_AA)

    cv2.imshow("Result", imgOrignal)
    cv2.waitKey(1)

