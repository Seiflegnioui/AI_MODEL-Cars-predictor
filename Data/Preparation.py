# Preparation.py :
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from Data.Encoding import Encoding
from Data.Graphics import Graphics
from sklearn.preprocessing import StandardScaler, RobustScaler,MinMaxScaler
from sklearn.linear_model import LogisticRegression
from typing import Optional
from sklearn.model_selection import train_test_split

class Preparation:
    _instance: Optional['Preparation'] = None

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.test_df = None
        self.__encoder = Encoding(self.df)
        self.__grapher = Graphics()
        self.standerizers = {}

    @classmethod
    def initializer(cls, df: pd.DataFrame):
        if cls._instance is None:
            cls._instance = cls(df)

    @classmethod
    def get_instance(cls) -> 'Preparation':
        if cls._instance is None:
            raise ValueError("Preparation is not initialized. Call initializer(df) first.")
        return cls._instance

    def getEndoder(self):
        return self.__encoder
    
    def set_df(self,df: pd.DataFrame):
        self.df = df

    def splitData(self):
        X = self.df.drop(columns=["Prix"])
        Y = self.df["Prix"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        self.df = pd.concat([X_train, y_train], axis=1)
        self.test_df = pd.concat([X_test, y_test], axis=1)

     
            
        
    def convert_annee_to_int(self):
        def process_annee(value):
            if pd.isna(value):  
                return np.nan
            str_value = str(value)
            number = "".join(ch for ch in str_value if ch.isdigit())
            return float(number) if number else np.nan  
        
        self.df["Année-Modèle"] = self.df["Année-Modèle"].map(process_annee).astype('float64')
        
        
    def show_heatmap(self,df):
        missing_matrix = df.isnull().astype(int)

        plt.figure(figsize=(10, 6))
        plt.imshow(missing_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        
        plt.colorbar(label='Missing (1) or Not Missing (0)')
        plt.title('Missing Values Heatmap')
        plt.xlabel('Row Index')
        plt.ylabel('Columns')
        plt.yticks(ticks=range(len(df.columns)), labels=df.columns)

        plt.show()
        
    def clean_null_prices(self):
        self.df = self.df.dropna(subset=['Prix'],axis=0)  
    
    def remove_null_rows(self):
        self.df = self.df[self.df.isnull().sum(axis=1) < 3]
        
    def visualize_null(self):
        null_counts = self.df.isnull().sum()
        
        plt.figure(figsize=(10, 5))
        null_counts.plot(kind='bar', color='coral')
        
        plt.title('Null Values Count per Column')
        plt.xlabel('Columns')
        plt.ylabel('Number of Null Values')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        

    def impute_portes_using_mode(self):
        mode_imputer = SimpleImputer(strategy='most_frequent')
        portes_values = self.df[["Nombre de portes"]].values  
        imputed_values = mode_imputer.fit_transform(portes_values)
        scaler = MinMaxScaler()
        normlized_values = scaler.fit_transform(imputed_values)
        self.df.loc[:, "Nombre de portes"] = normlized_values.flatten()

        self.standerizers["portes standarizer"] = scaler
  
        
    def impute_origine_using_mode(self):
        
        mode_imputer = SimpleImputer(strategy='most_frequent')
        origine_values = self.df[["Origine"]].values  
        imputed_values = mode_imputer.fit_transform(origine_values)
        self.df.loc[:, "Origine"] = imputed_values.flatten()  
        

    def impute_etat(self):
        excluded = [ 'Type de carburant', "Origine", "Première main", 
                'Puissance fiscale', "Nombre de portes", "Carburant_Diesel", 
                "Carburant_Electrique","Carburant_Essence","Carburant_Hybride","Carburant_LPG"]
        
        cols_for_imputation = [col for col in self.df.columns if col not in excluded]

        X = self.df[cols_for_imputation]
        y = self.df['État']
        
        testing_mask = y.isna()
        training_mask = ~y.isna()
        
        if testing_mask.sum() == 0:
            return  # Nothing to do
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X[training_mask], y[training_mask])
        
        predicted = model.predict(X[testing_mask])
        self.df.loc[testing_mask, "État"] = predicted
        print(f"✅ Imputed {testing_mask.sum()} missing values in 'État'.")

        
    def impute_pm_using_lr(self):
        excluded = [ 'Type de carburant', "Origine", "Première main", 
                'Puissance fiscale', "Nombre de portes", "Carburant_Diesel", 
                "Carburant_Electrique","Carburant_Essence","Carburant_Hybride","Carburant_LPG"]
        features = [col for col in self.df.columns if col not in excluded]
        
        X = self.df[features]
        y = self.df['Première main']
        
        train_mask = ~y.isna()
        test_mask = y.isna()
        
        model = LogisticRegression()  
        model.fit(X[train_mask], y[train_mask])
        
        self.df.loc[test_mask, 'Première main'] = model.predict(X[test_mask])
    
 
    def standardize_annee(self):
        
        self.df.loc[:, "Année-Modèle"] = self.df["Année-Modèle"].astype('float64')
        scaler = StandardScaler()
        self.df.loc[:, "Année-Modèle"] = scaler.fit_transform(
            self.df[["Année-Modèle"]]
        ).flatten()
        
        self.standerizers["year standarizer"] = scaler

    def standardize_kilometrage(self):
        
        self.df.loc[:, "Kilométrage"] = self.df["Kilométrage"].astype(float)
        scaler = RobustScaler()
        self.df.loc[:, "Kilométrage"] = scaler.fit_transform(
            self.df[["Kilométrage"]]
        ).flatten()
        
        self.standerizers["Kilométrage standarizer"] = scaler


    def standardize_prix(self):
        if (self.df["Prix"] < 0).any():
            raise ValueError("Prices cannot be negative for log scaling")
        
        self.df.loc[:, "Prix"] = np.log1p(self.df["Prix"].astype(float))
        scaler = StandardScaler()
        self.df.loc[:, "Prix"] = scaler.fit_transform(
            self.df[["Prix"]]
        ).flatten()
        
        self.standerizers["Prix standarizer"] = scaler

    def standardize_puissance_fiscale(self):
        self.df['Puissance fiscale'] = self.df['Puissance fiscale'].astype(float)
        scaler = StandardScaler()
        self.df['Puissance fiscale'] = scaler.fit_transform(
            self.df[['Puissance fiscale']]
        ).flatten()
        
        self.standerizers["Puissance fiscale scaler"] = scaler
        
    def standrize_etat(self):
        self.df = self.__encoder.standrize_etat(self.df)
    
    def encode_marque(self):
        self.df = self.__encoder.encode_marque(self.df)
    
    def encode_kilometrage(self):
        self.df = self.__encoder.encode_kilometrage(self.df)
     
    def encode_modele(self):
        self.df = self.__encoder.encode_modele(self.df)

    def encode_etat(self):
        self.df = self.__encoder.encode_etat(self.df)

    def encode_boite_de_vitesses(self):
        self.df = self.__encoder.encode_boite_de_vitesses(self.df)
        
    def encode_Type_de_carburant(self):
        self.df = self.__encoder.encode_Type_de_carburant(self.df)
          
    def encode_pm(self):
        self.df = self.__encoder.encode_pm(self.df)
        
    def encode_origine(self):
        self.df = self.__encoder.encode_origine(self.df)

    def encode_puissance_fiscale(self):
        self.df = self.__encoder.encode_puissance_fiscale(self.df)

    def show_distribution(self,column):
        self.__grapher.show_distribution(column,self.df)
    
    def get_test_data(self):
        return self.test_df
    
    def get_traib_data(self):
        return self.df