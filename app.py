from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open("ridge.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values from form in exact same order as training
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        DC = float(request.form['DC'])
        ISI = float(request.form['ISI'])
        BUI = float(request.form['BUI'])

        # Prepare array in same feature order used for training
        input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI]])

        # Scale inputs
        scaled_data = scaler.transform(input_data)

        # Predict using Ridge model
        prediction = ridge_model.predict(scaled_data)[0]

        return render_template('home.html', prediction_text=f"Predicted FWI Value: {prediction:.2f}")

    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
