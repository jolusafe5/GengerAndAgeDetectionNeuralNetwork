#Importamos las librerias para procesamiento de imagenes
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

IMG_DIMS=(96,96,3)

def convertImageToSingleArray (image):
    image=image.reshape(1,IMG_DIMS[0]*IMG_DIMS[1]*IMG_DIMS[2])
    return image


#Cargamos el Modelo de Detecciòn de Gènero que generamos.
model = load_model ('gender_detection.model')

#Activamos la càmara web
webcam = cv2.VideoCapture(0)

classes = ['man','woman']

#Recorremos los frames en un loop
while (webcam.isOpened()):
    #leemos el frame de la webcam
    status,frame = webcam.read()
    
    #Usamos la utileria para la detecciòn de caras
    face, confidence = cv.detect_face(frame)
    
    #Recorremos en un loop las caras detectadas
    for idx,f in enumerate(face):
        
        # obtenemos los puntos de la esquina del rectangulo de la cara
        (startX,startY) = f[0],f[1]
        (endX,endY) = f[2],f[3]
        
        cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2) #Dibujamos un rectangulo sobre la cara detectada
        
        face_crop = np.copy(frame[startY:endY,startX:endX]) #Recortamos la seccion de la cara identificada
        #<>
        if ((face_crop.shape[0])<10 or (face_crop.shape[1])<10):
            continue
            
        #preprocessing for gender detection model
        face_crop=cv2.resize(face_crop,(96,96))
        face_crop=face_crop.astype("float")/255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop,axis=0)
        
        #Se aplica la detecciòn de gènero sobre la cara detectada
        conf = model.predict(convertImageToSingleArray(face_crop))[0] # la predicciòn va  a regresar una matriz 2D
        
        #Se obtiene la etiqueta con el accuracy màximo
        idx = np.argmax(conf)
        label = classes[idx]
        
        identificationLabel= "{}: {:.2f}%".format(label,conf[idx]*100)
        
        Y=startY -10 if (startY -10 > 10 ) else (startY + 10)
        
        #Se escribe la etiqueta y el accuracy sobre el rectangulo de la cara detectada
        cv2.putText(frame, label,(startX,Y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        
    #Se muestra la salida
    cv2.imshow("Detecciòn de Gènero", frame)
    cv2.waitKey(1)
    
    #Para salir presionamos Q
    #if (cv2.waitKey(1)& 0xFF == ord('q')):
     #   break
        
    #Liberamos la conexiòn a la càmara que abrimos
    #webcam.release()
    #cv2.destroyAllWindows()
