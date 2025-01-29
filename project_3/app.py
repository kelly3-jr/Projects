from flask import Flask, request, jsonify
from sqlalchemy import create_engine
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask app
app = Flask(__name__)

# Connect to PostgreSQL database
engine = create_engine('postgresql://postgres:kelly%402003@localhost:5432/credi_card')

# Load your dataset (fraud_detection.py code) or any necessary ML model here
df = pd.read_csv('creditcard.csv')  # Example dataset

# Prepare the model (or load a pre-trained model)
X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Target variable
model = RandomForestClassifier()
model.fit(X, y)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.get_json()  # Assuming data is sent as JSON
        input_data = pd.DataFrame([data])  # Convert to DataFrame for model prediction

        # Predict using the model
        prediction = model.predict(input_data)

        # Return the result as JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
