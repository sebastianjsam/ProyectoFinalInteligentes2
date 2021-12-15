import io
from PIL import Image

import cv2
import requests
import json
import base64


if __name__=='__main__':
    url="http://127.0.0.1:5000/predict"


    image = open('imagen_0.jpg', 'rb')  # open binary file in read mode
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
    print(type(image_64_encode))
    print("tipo", base64.decodestring(image_64_encode))
    ##cv2.imwrite("sebast.png", image)

    #img = Image.open(io.BytesIO(image_read))
    #img.save("models.png")


    print(str(image_read))
    payload={ "id_Client": "0123123", "images": [ { "id":"1", "content": str(image_64_encode) } ], "models": [ "a","b" ] }
    response=requests.post(url,json=payload)

    if response.status_code==200:
        print(json.loads(response.content))
    else:
        print(response.status_code,response.content)