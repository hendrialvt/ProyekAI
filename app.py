import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model.h5')
scalar = joblib.load('scaling.pkl')

# Define label mapping
label_mapping = {
    0: 'GALAXY',
    1: 'QSO',
    2: 'STAR'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output_probabilities = model.predict(new_data)
    predicted_class = np.argmax(output_probabilities)
    predicted_label = label_mapping.get(predicted_class)
    return jsonify({
        'probabilities': output_probabilities.tolist(),
        'predicted_class': predicted_label
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    output_probabilities = model.predict(final_input)
    predicted_class = np.argmax(output_probabilities)
    predicted_label = label_mapping.get(predicted_class)
    return render_template("home.html", prediction_text="Tipe Stellar adalah {}".format(predicted_label))

if __name__ == "__main__":
    app.run(debug=True)
