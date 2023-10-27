import cv2
from os import path, makedirs
from imutils import resize 
from time import sleep as sl 


PersonName = input("Nombre del nuevo registro: ")


dataPath = "C:/Users/Jaime Farrel/Desktop/edwin/makerspace/proyects/reconocimiento facial/data"

emotionsPath = dataPath + '/' + PersonName

if not path.exists(emotionsPath):
    print('carpeta creada: ',emotionsPath)
    makedirs(emotionsPath)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0 
while True:
    ret, frame = cap.read()
    if ret == False:break 
    frame = resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(frame,3.1,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(gray, (x,y),(x+w,y+h),(0,255,0),2)
        rostro = cv2.cvtColor(auxFrame[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        rostro = cv2.resize(rostro, (150,160),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(emotionsPath + '/rostro_{}.jpg'.format(count),rostro)
        print('imagen tomada_{}'.format(count))

        count = count  + 1
    cv2.imshow('Recoleccion de Datos',gray)

    k = cv2.waitKey(1) 
    if k == 27 or count >= 50:
        break 


cap.release()
cv2.destroyAllWindows()