import cv2 
from os import listdir
import numpy as np 

#escoges la carpeta, la conviertes en una lista y la muestras en la terminal

dataPath = "C:/Users/Jaime Farrel/Desktop/edwin/makerspace/proyects/reconocimiento facial/data"

peopleList = listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('procesando')
    
    for fileName in listdir(personPath): 
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        # image = cv2.imread(personPath+'/'+fileName,0)
        # cv2.imshow('image',image)
        # cv2.waitKey(10)
    label = label + 1
# print('labels= ',labels)
# print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
# print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))


#Formas de entrenar el modelo 


# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
count =0 
# list_models = listdir('./LBPH')

# print( str(list_models)) print the result of the read of file LBPH

#train data: 

#ejemplos para agregar en el codigo de abajo: ('modeloEigenFace') ('modeloFisherFace') ('modeloLBPH')
print('preparando el modelo...')
face_recognizer.train(facesData, np.array(labels))

#save the modelo 
face_recognizer.write('C:/Users/Jaime Farrel/Desktop/edwin/makerspace/proyects/reconocimiento facial/LBPH/modeloLBPHFace_.xml')
print('!El modelo ha sido entrenado!')
