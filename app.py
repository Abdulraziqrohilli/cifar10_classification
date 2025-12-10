from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model("cifar10_cnn_model.h5")

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_text = ""
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file)
        img = img.resize((32,32))
        img = np.array(img) / 255.0
        if img.shape[-1] == 4:  # RGBA -> RGB
            img = img[...,:3]
        img = img.reshape(1,32,32,3)
        pred = model.predict(img)
        predicted_class = class_names[np.argmax(pred)]
        prediction_text = f"Predicted: {predicted_class}"
    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
