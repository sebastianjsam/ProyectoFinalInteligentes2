import numpy as np
import cv2
import metodosEnt
import io
from PIL import Image

import cv2
import requests
import json
import base64

nameWindow ="Datos de configuración"

def nothing(x):
    pass

def construirVentana():
    cv2.namedWindow(nameWindow)
    cv2.createTrackbar("min",nameWindow,0,255,nothing)
    cv2.createTrackbar("max",nameWindow,1,100,nothing)
    cv2.createTrackbar("kernel",nameWindow,0,255,nothing)
    cv2.createTrackbar("areaMin",nameWindow,500,10000,nothing)

cap=cv2.VideoCapture(1)
construirVentana()
img_counter=0
def calcularAreas(objetos):
    areas=[]
    for objetoActual in objetos:
        areas.append(cv2.contourArea(objetoActual))
    return areas

def redimensionar(img_name,borde,contornos):
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)
        if w>50 and h>50:
            nuevaimagen = borde[y:y+h,x:x+w]
            cv2.imwrite(img_name, nuevaimagen)



def detectarForma(imagen,variable):
    imagenGris=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grises",imagenGris)
    min=cv2.getTrackbarPos("min",nameWindow)
    max=cv2.getTrackbarPos("max",nameWindow)
    bordes=cv2.Canny(imagenGris,min,max)
    tamañokernel=cv2.getTrackbarPos("kernel",nameWindow)
    kernel=np.ones((tamañokernel,tamañokernel),np.uint8)
    bordes=cv2.dilate(bordes,kernel)
    cv2.imshow("Bordes",bordes)
    objetos,jerarquias=cv2.findContours(bordes,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    nuevaImagen = np.zeros_like(bordes)
    cv2.drawContours(nuevaImagen, objetos, -1, 255, 1)
    cv2.imshow('contornos', nuevaImagen)
    areas=calcularAreas(objetos)
    i=0
    areaMin=cv2.getTrackbarPos("areaMin",nameWindow)

    #contornos
           # ret,th=cv2.threshold(frame,200,255,cv2.THRESH_BINARY)
            #contorno,jerarquia=cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #for i in range(len(contorno)):
     #   cv2.draeContours(frame,contorno,i,(0,255,0),3)
      #  print("c")

    for objetoActual in objetos:
        if areas[i]>=areaMin:
            vertices=cv2.approxPolyDP(objetoActual,0.025*cv2.arcLength(objetoActual, closed=True),True)
            if len(vertices) == 4:
                x, y, w, h = cv2.boundingRect(vertices)
                new_img=frame[y:y+h,x:x+w]
                #cv2.imwrite(img_name,new_img)
                print("objecto actual",objetoActual)
                cv2.drawContours(frame, [vertices], -1, (0, 255, 0), 3)
        i = i+1

    return [bordes,objetos]

def conectarBack():
    url = "http://127.0.0.1:5000/predict"

    image = open('imagen_0.jpg', 'rb')  # open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
    print(type(image_64_encode))
    print("tipo", base64.decodestring(image_64_encode))
    ##cv2.imwrite("sebast.png", image)

    # img = Image.open(io.BytesIO(image_read))
    # img.save("models.png")

    print(str(image_read))
    payload = {"id_Client": "0123123", "images": [{"id": "1", "content": str(image_64_encode)}], "models": ["a", "b"]}
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        print(json.loads(response.content))
    else:
        print(response.status_code, response.content)








#______________________________________________
acum = 0
cont = 0
variable = False
print("inicio")
while(True):
    ret,frame = cap.read()
    cv2.imshow('Video1',frame)
    borde,contorno=detectarForma(frame,variable)
    if not ret:
        break
    k=cv2.waitKey(1)
    if k%256 ==99: #deteccion del cuadrado
        variable=True
        print("tecla c")
    if k%256 == 101:
        cont += 1
        img_name ="imagen_{}.jpg".format(img_counter)
        redimensionar(img_name,borde,contorno)
        detectarForma(frame,img_name)
        img_counter += 1
        acum = acum + metodosEnt.probarModelo(img_name)

        metodosEnt.mostrarAcumulado(acum, img_name)
        print("backen")
        conectarBack()









cap.release
####### Implementacion del Modelo
#
print("El acumulado es: ", acum)

cv2.destroyAllWindows()