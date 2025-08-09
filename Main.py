import pandas as pd
from Data.Preparation import Preparation
import joblib
from eda import remove_outliers_iqr

colonnes_cibles = [
        "Prix", "Année-Modèle", "Kilométrage",
        "Nombre de portes", "Puissance fiscale", "État"
    ]
df = pd.read_csv('voitures_avito.csv')

def main(df):
    Preparation.initializer(df)
    preparater =   Preparation.get_instance()
    
    preparater.convert_annee_to_int()
    preparater.clean_null_prices()
    preparater.remove_null_rows()
    preparater.impute_portes_using_mode()
    preparater.impute_origine_using_mode()
    
    preparater.encode_marque()
    preparater.encode_kilometrage()
    preparater.encode_modele()
    preparater.encode_etat()
    preparater.encode_origine()
    preparater.encode_pm()
    preparater.encode_boite_de_vitesses()
    preparater.encode_Type_de_carburant()
    preparater.encode_puissance_fiscale()

    preparater.set_df(remove_outliers_iqr(preparater.get_traib_data(), colonnes_cibles)[0])
    preparater.splitData()

    preparater.standardize_prix()
    preparater.standardize_annee()
    preparater.standardize_kilometrage()
    preparater.standrize_etat()
    preparater.impute_etat()
    preparater.impute_pm_using_lr()
    preparater.standardize_puissance_fiscale()
    
    
    train_df= preparater.get_traib_data()
    test_df = preparater.get_test_data()

    print(train_df.isnull().sum().sum())
    print(test_df.isnull().sum().sum())
    preparater.show_heatmap(test_df)
    train_df.to_csv('voitures_train.csv', index=False)
    test_df.to_csv('voitures_test.csv', index=False)
    joblib.dump(preparater, 'preparation_instance.pkl')
    print(train_df)
    print(test_df)

main(df)
