import numpy as np
import cv2
import metodosEnt

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



def detectarForma(imagen):
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

    for objetoActual in objetos:
        if areas[i]>=areaMin:

            vertices=cv2.approxPolyDP(objetoActual,0.025*cv2.arcLength(objetoActual, closed=True),True)

            if len(vertices) == 4 :
                x, y, w, h = cv2.boundingRect(vertices)
                new_img=frame[y:y+h,x:x+w]
                #cv2.imwrite(img_name,new_img)
        i = i+1

    return [bordes,objetos]
#______________________________________________
acum = 0
cont = 0
print("inicio")
while(True):
    ret,frame = cap.read()
    cv2.imshow('Video1',frame)
    borde,contorno=detectarForma(frame)
    if not ret:
        break
    k=cv2.waitKey(1)
    if k%256 == 112:
        cont += 1
        img_name ="imagen_{}.jpg".format(img_counter)
        redimensionar(img_name,borde,contorno)
        #detectarForma(frame,img_name)
        img_counter += 1
        acum = acum + metodosEnt.probarModelo(img_name)

        metodosEnt.mostrarAcumulado(acum, img_name)

cap.release
####### Implementacion del Modelo
#
print("El acumulado es: ", acum)

cv2.destroyAllWindows()