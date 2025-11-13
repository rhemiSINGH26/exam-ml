from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)
model, acc = joblib.load('forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'], data['fbs'], data['thalach']]).reshape(1, -1)
    prediction = model.predict(features)[0]
    result = 'Yes' if prediction == 1 else 'No'
    return jsonify({
        'Prediction': result,
        'Model Accuracy': round(acc, 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)