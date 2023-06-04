import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model.h5')
scalar= joblib.load('scaling.pkl')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    return jsonify(output[0].tolist())

if __name__ == "__main__":
    app.run(debug=True)
