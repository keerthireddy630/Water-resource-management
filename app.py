from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from index.html

# Load dataset and train model at startup
data = pd.read_csv(r"C:\Users\rajes\Downloads\water_data.csv")

X = data[['Temperature_C', 'Precipitation_mm', 'Water_Demand_MLD']]
y = data['Groundwater_Level_m']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    temp = data['temperature']
    precip = data['precipitation']
    demand = data['water_demand']

    features = pd.DataFrame({
        'Temperature_C': [temp],
        'Precipitation_mm': [precip],
        'Water_Demand_MLD': [demand]
    })

    prediction = model.predict(features)[0]

    return jsonify({'predicted_groundwater_level': round(float(prediction), 2)})

if __name__ == '__main__':
    app.run(debug=True)
