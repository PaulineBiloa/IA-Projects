# ====================================================================
# FICHIER : app.py
# API Flask pour la prediction des especes d'iris
# ====================================================================

from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np
import pandas as pd

# Creer l'application Flask
app = Flask(__name__)

# Charger le modele et le scaler au demarrage
print("Chargement du modele...")
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model_info.pkl', 'rb') as file:
    model_info = pickle.load(file)

print("Modele charge avec succes !")
print(f"Modele : {model_info['model_name']}")
print(f"Exactitude : {model_info['accuracy']*100:.2f}%")

# ====================================================================
# ROUTE 1 : PAGE D'ACCUEIL
# ====================================================================

@app.route('/')
def home():
    """Page d'accueil avec formulaire de test"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Iris - Classification</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #FFF5FA 0%, #E6B3FF 100%);
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(139, 71, 137, 0.3);
            }
            h1 {
                color: #8B4789;
                text-align: center;
                border-bottom: 3px solid #E6B3FF;
                padding-bottom: 10px;
            }
            .info {
                background: #FFF5FA;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #E6B3FF;
            }
            .form-group {
                margin: 15px 0;
            }
            label {
                display: block;
                color: #8B4789;
                font-weight: bold;
                margin-bottom: 5px;
            }
            input {
                width: 100%;
                padding: 10px;
                border: 2px solid #E6B3FF;
                border-radius: 5px;
                font-size: 16px;
            }
            button {
                width: 100%;
                padding: 12px;
                background: #E6B3FF;
                color: #8B4789;
                border: none;
                border-radius: 5px;
                font-size: 18px;
                font-weight: bold;
                cursor: pointer;
                margin-top: 10px;
            }
            button:hover {
                background: #D4A5D4;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 10px;
                display: none;
            }
            .success {
                background: #d4edda;
                border: 2px solid #28a745;
                color: #155724;
            }
            .routes {
                margin-top: 30px;
                padding: 15px;
                background: #FFF5FA;
                border-radius: 10px;
            }
            code {
                background: #f8f9fa;
                padding: 2px 6px;
                border-radius: 3px;
                color: #C71585;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌸 API de Classification des Iris 🌸</h1>
            
            <div class="info">
                <h3 style="margin-top:0; color:#8B4789;">ℹ️ Informations du modèle</h3>
                <p><strong>Modèle utilisé :</strong> {{ model_name }}</p>
                <p><strong>Exactitude :</strong> {{ accuracy }}%</p>
                <p><strong>Date d'entraînement :</strong> {{ training_date }}</p>
                <p><strong>Espèces :</strong> Setosa, Versicolor, Virginica</p>
            </div>

            <h2 style="color:#8B4789;">🧪 Formulaire de test</h2>
            <form id="predictionForm">
                <div class="form-group">
                    <label>Longueur du Sépale (cm) :</label>
                    <input type="number" step="0.1" name="sepal_length" 
                           placeholder="Ex: 5.1" required>
                </div>
                <div class="form-group">
                    <label>Largeur du Sépale (cm) :</label>
                    <input type="number" step="0.1" name="sepal_width" 
                           placeholder="Ex: 3.5" required>
                </div>
                <div class="form-group">
                    <label>Longueur du Pétale (cm) :</label>
                    <input type="number" step="0.1" name="petal_length" 
                           placeholder="Ex: 1.4" required>
                </div>
                <div class="form-group">
                    <label>Largeur du Pétale (cm) :</label>
                    <input type="number" step="0.1" name="petal_width" 
                           placeholder="Ex: 0.2" required>
                </div>
                <button type="submit">🔮 Prédire l'espèce</button>
            </form>

            <div id="result" class="result"></div>

            <div class="routes">
                <h3 style="color:#8B4789;">📡 Routes API disponibles</h3>
                <p><strong>POST /predict</strong> - Prédire une espèce</p>
                <p><strong>GET /info</strong> - Informations sur le modèle</p>
                <p><strong>GET /health</strong> - Vérifier l'état de l'API</p>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').onsubmit = async (e) => {
                e.preventDefault();
                
                const formData = new FormData(e.target);
                const data = {
                    sepal_length: parseFloat(formData.get('sepal_length')),
                    sepal_width: parseFloat(formData.get('sepal_width')),
                    petal_length: parseFloat(formData.get('petal_length')),
                    petal_width: parseFloat(formData.get('petal_width'))
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });

                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (result.success) {
                        resultDiv.className = 'result success';
                        resultDiv.style.display = 'block';
                        resultDiv.innerHTML = `
                            <h3 style="margin-top:0;">✅ Prédiction réussie !</h3>
                            <p><strong>Espèce prédite :</strong> ${result.prediction}</p>
                            <p><strong>Probabilités :</strong></p>
                            <ul>
                                ${result.probabilities.map(p => 
                                    `<li>${p.species}: ${p.probability}%</li>`
                                ).join('')}
                            </ul>
                        `;
                    } else {
                        resultDiv.className = 'result';
                        resultDiv.style.background = '#f8d7da';
                        resultDiv.style.border = '2px solid #dc3545';
                        resultDiv.style.display = 'block';
                        resultDiv.innerHTML = `<p>❌ Erreur: ${result.error}</p>`;
                    }
                } catch (error) {
                    alert('Erreur de connexion: ' + error);
                }
            };
        </script>
    </body>
    </html>
    """
    return render_template_string(html, 
                                 model_name=model_info['model_name'],
                                 accuracy=f"{model_info['accuracy']*100:.2f}",
                                 training_date=model_info['training_date'])

# ====================================================================
# ROUTE 2 : PREDICTION
# ====================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour faire des predictions
    
    Input JSON:
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    Output JSON:
    {
        "success": true,
        "prediction": "setosa",
        "probabilities": [...]
    }
    """
    try:
        # Recuperer les donnees JSON
        data = request.get_json()
        
        # Valider que toutes les caracteristiques sont presentes
        required_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    'success': False,
                    'error': f'Caracteristique manquante : {feature}'
                }), 400
        
        # Creer un tableau numpy avec les caracteristiques
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        
        # Normaliser les caracteristiques
        features_scaled = scaler.transform(features)
        
        # Faire la prediction
        prediction = model.predict(features_scaled)[0]
        
        # Obtenir les probabilites si disponible
        probabilities = []
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_scaled)[0]
            probabilities = [
                {'species': species, 'probability': f'{prob*100:.2f}'}
                for species, prob in zip(model_info['species'], proba)
            ]
        
        # Retourner la reponse
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probabilities': probabilities,
            'input_features': {
                'sepal_length': data['sepal_length'],
                'sepal_width': data['sepal_width'],
                'petal_length': data['petal_length'],
                'petal_width': data['petal_width']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ====================================================================
# ROUTE 3 : INFORMATIONS SUR LE MODELE
# ====================================================================

@app.route('/info', methods=['GET'])
def info():
    """Retourner les informations sur le modele"""
    return jsonify({
        'model_name': model_info['model_name'],
        'accuracy': f"{model_info['accuracy']*100:.2f}%",
        'features': model_info['features'],
        'species': model_info['species'],
        'training_date': model_info['training_date']
    })

# ====================================================================
# ROUTE 4 : HEALTH CHECK
# ====================================================================

@app.route('/health', methods=['GET'])
def health():
    """Verifier que l'API fonctionne"""
    return jsonify({
        'status': 'healthy',
        'message': 'API is running properly'
    })

# ====================================================================
# ROUTE 5 : PREDICTIONS MULTIPLES
# ====================================================================

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Faire des predictions pour plusieurs echantillons a la fois
    
    Input JSON:
    {
        "samples": [
            {"sepal_length": 5.1, "sepal_width": 3.5, ...},
            {"sepal_length": 6.2, "sepal_width": 2.8, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({
                'success': False,
                'error': 'Aucun echantillon fourni'
            }), 400
        
        # Preparer les caracteristiques
        features_list = []
        for sample in samples:
            features_list.append([
                sample['sepal_length'],
                sample['sepal_width'],
                sample['petal_length'],
                sample['petal_width']
            ])
        
        features = np.array(features_list)
        features_scaled = scaler.transform(features)
        
        # Faire les predictions
        predictions = model.predict(features_scaled)
        
        # Retourner les resultats
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'sample_index': i,
                'prediction': pred,
                'input': samples[i]
            })
        
        return jsonify({
            'success': True,
            'total_samples': len(samples),
            'predictions': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ====================================================================
# LANCEMENT DE L'APPLICATION
# ====================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DEMARRAGE DE L'API FLASK")
    print("=" * 60)
    print(f"Modele : {model_info['model_name']}")
    print(f"Exactitude : {model_info['accuracy']*100:.2f}%")
    print("\nRoutes disponibles :")
    print("  - GET  /         : Page d'accueil")
    print("  - POST /predict  : Prediction simple")
    print("  - POST /predict_batch : Predictions multiples")
    print("  - GET  /info     : Informations du modele")
    print("  - GET  /health   : Health check")
    print("\nServeur demarre sur : http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    
    import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render fournit PORT
    app.run(debug=True, host='0.0.0.0', port=port)
