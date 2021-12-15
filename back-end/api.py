import base64
import io
from PIL import Image
from flask import Flask
from flask_restful import Resource, Api
##import twitter_tools as tt


from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource

import metodosEnt

app = Flask(__name__)
api = Api(app)

class Hello(Resource):
    contador=0
    def get(self, name):
        prediccion1=self.predicciones()
        prediccion2=self.predicciones()

        return {"state":"success",
                "message":"predicciones realizadas exitosamente",
                "results":[{"prediccion":prediccion1},{"prediccion":prediccion2}]}
        ##return {"Hello":name}

    def predicciones(self):
        self.contador+=1
        return self.contador
@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return f'Post {post_id}'



class Search(Resource):
    def get(self, img):
        ##person = tt.twitter(name)
        lista=eval(img)
        return lista


class Predict(Resource):
    def post(self):
        print("request ", request.get_json())
        json_data = request.get_json()
        return self.procesarDatos(json_data,"1")

    def procesarDatos(self, json_data,id):
        img_name = "imagen_1.jpg"
        modelo1=metodosEnt.probarModelo(img_name)
        print("2?",modelo1)

        listPrincipal=[]
        list=[]
        print("models", json_data)

        for e in json_data["models"]:
            for x in json_data["images"]:
                print("base 64", x["content"])
                img_byte=x["content"]
                print("img byte",img_byte)
                t=bytes(img_byte, encoding="raw_unicode_escape")
                print("tipo",type(t))
                #img_decode=base64.decodestring(t)
               # print("decode",base64.b64decode(t, 'b/'))
                ##print(base64.decodestring(img_byte))
                #img = Image.open(io.BytesIO(t))
                #img.save("models.png")
                modelo1 = metodosEnt.probarModeloEspecifico(img_name,e)
                list.append({
                   "class": str(modelo1),
                   "id-image": x["id"]})
            listPrincipal.append({"model_id":e,"result":list})
            list=[]

        return self.respuesta(listPrincipal)


    def respuesta(self,list):
        json = {"state": "success", "result": list}
        return json

api.add_resource(Predict, '/predict')

api.add_resource(Search, '/prediccion/<img>')
api.add_resource(Hello, '/hello/<name>')

if __name__ == '__main__':
    app.run(debug=True)