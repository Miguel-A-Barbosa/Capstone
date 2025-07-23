import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib

from ajustar_datos import transformador

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

modelo = joblib.load('modelo_regresion.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    datos = request.get_json()

    valores = transformador(datos['valores'])

    valores_escalados = scaler.transform(valores)

    prediccion = modelo.predict(valores_escalados)

    return jsonify({'Costos_MillonesCOP': round(float(prediccion[0]), 3)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)