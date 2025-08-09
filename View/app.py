import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for
import joblib
import pandas as pd
import numpy as np
# import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.Preparation import Preparation

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, "..", "voitures_test.csv")


current_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_dir, "..", "Model", "random_forest_model.pkl")

if os.path.exists(model_path):
    print(f"Model file found at: {model_path}")
else:
    print(f"Error: Model file not found at {model_path}")
    
df = pd.read_csv(file_path)
df = df.drop(columns=["Prix"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        
        marque_columns = [col for col in df.columns.to_list() if col.startswith("Marque")]
        modele_columns = [col for col in df.columns.to_list() if col.startswith("Modèle")]
        carburant_columns = [col for col in df.columns.to_list() if col.startswith("Carburant")]
        origine_columns = [col for col in df.columns.to_list() if col.startswith("Origine")]

        data_dict = form_data.to_dict()
        processed_data = {
            'Année-Modèle': int(data_dict.get('Année-Modèle')),
            'Boite de vitesses': int(data_dict.get('Boite de vitesses')),
            'Type de carburant': data_dict.get('Type de carburant'),
            'Kilométrage': float(data_dict.get('Kilométrage')),
            'Marque': data_dict.get('Marque'),
            'Modèle': data_dict.get('Modèle'),
            'Nombre de portes': int(data_dict.get('Nombre de portes')),
            'Origine': data_dict.get('Origine'),
            'Première main': int(data_dict.get('Première main')),
            'Puissance fiscale': int(data_dict.get('Puissance fiscale')),
            'État': int(data_dict.get('État'))
        }

        preparation_path = os.path.join(current_dir, "..", "preparation_instance.pkl")

        preparator = joblib.load(preparation_path)
        model = joblib.load(model_path)
        
        scaled_year = None
        scaled_kilometrage = None
        scaled_pf = None
        scaled_etat = None
        scaled_portes = None


        origine_one_hot = [1 if processed_data['Origine'] == col.split("Origine_")[-1] else 0 for col in origine_columns]
        marque_one_hot = [1 if processed_data['Marque']==  col.split("Marque_")[-1] else 0 for col in marque_columns]
        modele_one_hot = [1 if processed_data['Modèle'] == col.split("Modèle_")[-1] else 0 for col in modele_columns]
        carburant_one_hot = [1 if processed_data['Type de carburant'] == col.split("Carburant_")[-1] else 0 for col in carburant_columns]
        
        df_origine = pd.DataFrame([origine_one_hot], columns=origine_columns)
        df_marque = pd.DataFrame([marque_one_hot], columns=marque_columns)
        df_modele = pd.DataFrame([modele_one_hot], columns=modele_columns)
        df_carburant = pd.DataFrame([carburant_one_hot], columns=carburant_columns)
        
        
        pd.concat([df_origine,df_marque,df_modele,df_carburant],axis=1)

        for key, scaler in preparator.standerizers.items():
            if str(key).startswith('year'):
                scaled_year = scaler.transform([[processed_data["Année-Modèle"]]])
            if str(key).startswith('Kilom'):
                scaled_kilometrage = scaler.transform([[processed_data["Kilométrage"]]])
            if str(key).startswith('Puissance'):
                scaled_pf = scaler.transform([[processed_data["Puissance fiscale"]]])  
            if str(key).startswith('portes'):
                scaled_portes = scaler.transform([[processed_data["Nombre de portes"]]])

        for key, scaler in preparator.getEndoder().standrizers.items():
            if str(key).startswith('etat'):
                scaled_etat = scaler.transform([[processed_data["État"]]])
                

        df_all = pd.concat([ df_marque, df_modele,df_origine, df_carburant], axis=1)

        scaled_data = {
            'Année-Modèle': scaled_year[0][0] if scaled_year is not None else None,
            'Boite de vitesses': processed_data['Boite de vitesses'],
            'Kilométrage': scaled_kilometrage[0][0] if scaled_kilometrage is not None else None,
            'Nombre de portes': scaled_portes[0][0] if scaled_portes is not None else None,
            'Première main': processed_data['Première main'],
            'Puissance fiscale': scaled_pf[0][0] if scaled_pf is not None else None,
            'État': scaled_etat[0][0] if scaled_etat is not None else None,
        }

        df_scaled = pd.DataFrame([scaled_data])
        final_input = pd.concat([df_scaled,df_all], axis=1)
        
        predicted_price = model.predict(final_input)
        predicted_price = preparator.standerizers["Prix standarizer"].inverse_transform(predicted_price.reshape(-1, 1))
        predicted_price = np.exp(predicted_price)
        print(predicted_price)

        return redirect(url_for('show', etat=processed_data['État'],carburant=processed_data['Type de carburant'] ,year = processed_data['Année-Modèle'] ,price=predicted_price[0][0], kilom = processed_data['Kilométrage'] ,name=processed_data['Marque'] + ' '+processed_data['Modèle'],vitesse=processed_data["Boite de vitesses"])) 
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/show')
def show():
    
    price = request.args.get('price')
    year = request.args.get('year')
    vitesse = request.args.get('vitesse')
    kilom = request.args.get('kilom')
    name = request.args.get('name')
    carburant = request.args.get('carburant')
    etat = request.args.get('etat')

    etat = "Neuf" if int(etat) == 0 else ("Excellent" if int(etat) == 1 else ('Très bon' if int(etat) == 2 else ("Bon" if int(etat) == 3 else ('Correct' if int(etat) == 4 else ('Pour Pièces' if int(etat) == 5 else ('Endommagé' if int(etat) == 6 else "Error"))))))
    
    return render_template("predict.html" ,name=name ,price=price,year=year,vitesse=vitesse,kilom=kilom,carburant=carburant,etat=etat )

if __name__ == '__main__':
    app.run(debug=True)
