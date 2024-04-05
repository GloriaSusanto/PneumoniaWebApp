import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__, template_folder='templates')

def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = img.convert("RGB")
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def pneumoniaPredict(image_path):
    # Load the model
    loaded_model = tf.keras.models.load_model('./asset/efficientnetb3-pneumonia-99.04-random.h5')

    # Get class labels
    class_labels = ['Normal', 'Pneumonia']

    # Load the image
    new_image = load_image(image_path)

    # Make predictions
    predictions = loaded_model.predict(new_image)

    # Ensure the predicted class index is within the range of class labels
    predicted_class_index = np.argmax(predictions[0])
    if predicted_class_index < len(class_labels):
        predicted_label = class_labels[predicted_class_index]
        probability = f'{predictions[0][predicted_class_index] * 100:.2f}%'
    else:
        # If the predicted class index is out of range, set predicted label to None and probability to 0%
        predicted_label = None
        probability = '0%'

    return predicted_label, probability

@app.route("/", methods=['GET'])
def index_get():
    return render_template('index.html', label=None, accuracy=None, file=None)

@app.route("/", methods=['POST'])
def index_post():
    try:
        imagefile = request.files['file']  # Use 'file' as the file field name
        image_path = "static/images/test.png"
        imagefile.save(image_path)

        predicted_label, probability = pneumoniaPredict(image_path)

        return render_template('index.html', label=predicted_label, accuracy=probability, file=image_path)
    except Exception as e:
        return render_template('error.html', error=e)

if __name__ == "__main__":
    app.run(debug=True)
