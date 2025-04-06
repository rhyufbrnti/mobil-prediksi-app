import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("toyota.csv")

# Preprocessing (pastikan kolom sesuai dataset-mu)
data = data.dropna()
data = data[data['price'] < 60000]  # Optional filter

# Fitur dan target
X = data[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']]
y = data['price']

# Tentukan kolom kategorikal dan numerik
categorical_cols = ['model', 'transmission', 'fuelType']
numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

# Buat encoder untuk kolom kategorikal
encoder = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

# Fit encoder ke data, lalu transform
X_encoded = encoder.fit_transform(X)

# Simpan encoder ke file
joblib.dump(encoder, 'model_encoder.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Buat dan latih model
model = LinearRegression()
model.fit(X_train, y_train)

# Simpan model ke file
joblib.dump(model, 'model_lr.pkl')

print("Model dan encoder berhasil disimpan ke model_lr.pkl dan model_encoder.pkl")