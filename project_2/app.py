from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('stroke_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json
    features = np.array([data['gender'], data['age'], data['hypertension'],
                         data['heart_disease'], data['ever_married'],
                         data['work_type'], data['Residence_type'],
                         data['avg_glucose_level'], data['bmi'], data['smoking_status']]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    return jsonify({'stroke': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
