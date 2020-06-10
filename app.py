from flask import Flask
import flask
from flask import Response, render_template
from preprocessing import load
from PIL import Image
import numpy as np
import io
import keras
from gevent.pywsgi import WSGIServer

import tensorflow as tf


xray_model = keras.models.load_model('model.model')


app = Flask(__name__, static_folder="static")

app.config['UPLOAD_FOLDER'] = "tmp_dir"


@app.route("/predict", methods=["POST"])
def predict():

    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("file"):
            # read the image in PIL format
            image = flask.request.files["file"].read()
            print(f"The image is: {flask.request.files['file']}")
            image = Image.open(io.BytesIO(image))
            # preprocess the image and prepare it for classification
            image_ = load(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            # model = keras.models.load_model('model.model')
            preds = xray_model.predict(image_)
            data["success"] = True

            preds_list = preds.tolist()[0]
            preds_list[0] = round(preds_list[0]*100,2)
            preds_list[1] = round(preds_list[1]*100,2)

            data["predictions"] = preds_list
            

    # return the data dictionary as a JSON response
    if flask.request.content_type.startswith("application/json"):
        return flask.jsonify(data)
    return render_template("result.html", data=data)



@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 5010), app)
    http_server.serve_forever()