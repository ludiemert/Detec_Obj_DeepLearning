import cv2

video = cv2.VideoCapture(0) #abre a webcam

amostra = 1 #para sempre criar um arquivo novo

#loop para capturar fotos, qdo aperta s
while True:
    check,img = video.read()

    if cv2.waitKey(1) & 0XFF ==ord('s'):
        imgR = cv2.resize(img,(32,32)) #tamanho img, tempo de processamento que o algortimo vai levar qto maior mais vai demorar, porque faz a multiplicacao de matriz eleva o processamento e o notebook
        #mesmo com img pqe em px conseg bons resultados
        cv2.imwrite(f'Imagens/1/im{amostra}.jpg',imgR) #troca a pasta p 0, 1 ou 2 para capturar img
        print(f'imagem salva{amostra}')
        amostra +=1 #qto maior diversidade da amostra melhor o resultado

    cv2.imshow('Captura', img)
    cv2.waitKey(1)