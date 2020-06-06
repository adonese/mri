from flask import Flask
import flask
from preprocessing import load
from PIL import Image
import numpy as np
import io
import keras
from gevent.pywsgi import WSGIServer

import tensorflow as tf


xray_model = keras.models.load_model('model.model')


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()

            print(flask.request.files["image"])


            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image_ = load(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            # model = keras.models.load_model('model.model')
            preds = xray_model.predict(image_)
            data["success"] = True
            data["predictions"] = preds.tolist()[0]

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5010), app)
    http_server.serve_forever()