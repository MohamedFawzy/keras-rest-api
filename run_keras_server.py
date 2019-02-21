from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io



app = flask.Flask(__name__)
model = None



def load_model():

    global model
    model = ResNet50(weights="imagenet")


def prepare_image(image , target):

    if image.mode != "RGB":
        image = image.convert("RGB")


    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return  image

@app.route("/predict", methods=["POST"])
def predict():

    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.file.get("image"):
            image = flask.request.file["image"].read()