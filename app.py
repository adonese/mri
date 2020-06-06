from flask import Flask
from .prepocessing import prepare_image
from PIL import Image
import numpy as np
import io

# pip install flask gevent requests pillow
model = keras.models.load_model('/content/drive/My Drive/model.model')
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
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            data["success"] = True
            data["predictions"] = preds

    # return the data dictionary as a JSON response
    return flask.jsonify(data)