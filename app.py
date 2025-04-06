from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model dan encoder
model = joblib.load('model_lr.pkl')
encoder = joblib.load('model_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    model_input = request.form['model']
    year = int(request.form['year'])
    transmission = request.form['transmission']
    mileage = int(request.form['mileage'])
    fuelType = request.form['fuelType']
    tax = int(request.form['tax'])
    mpg = float(request.form['mpg'])
    engineSize = float(request.form['engineSize'])

    # Buat DataFrame dari input
    input_df = pd.DataFrame([{
        'model': model_input,
        'year': year,
        'transmission': transmission,
        'mileage': mileage,
        'fuelType': fuelType,
        'tax': tax,
        'mpg': mpg,
        'engineSize': engineSize
    }])

    # Transformasi input menggunakan encoder
    input_encoded = encoder.transform(input_df)

    # Prediksi harga
    prediction = model.predict(input_encoded)[0]
    prediction = round(prediction, 2)

    return render_template('index.html', prediction_text=f'Perkiraan harga mobil: £{prediction}')

if __name__ == '__main__':
    app.run(debug=True)