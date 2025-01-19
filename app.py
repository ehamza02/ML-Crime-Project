from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
model = load_model('model/crime_model.h5')

# Directory for file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected", 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Process the Excel file
    try:
        data = pd.read_excel(file_path)
        # Ensure the data contains the expected columns (validate input)
        required_columns = ['Population', 'Median_Income']  # Add all required columns
        if not all(col in data.columns for col in required_columns):
            return "Invalid file format. Missing required columns.", 400

        # Prepare data for prediction
        features = data[required_columns].to_numpy()
        predictions = model.predict(features)

        # Add predictions to the DataFrame
        data['Violent_Crime_Prediction'] = predictions[:, 0]
        data['Nonviolent_Crime_Prediction'] = predictions[:, 1]

        # Save the results to a new file
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.xlsx')
        data.to_excel(output_path, index=False)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        print(e)
        return "Error processing file.", 500

if __name__ == '__main__':
    app.run(debug=True)
