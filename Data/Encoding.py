# Encoding.py :
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Encoding: 
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.standrizers = {}
        self.encoders = {}
    
    def encode_modele(self, df):
        self.df = df
        dummies = pd.get_dummies(self.df["Modèle"], prefix="Modèle")
        self.df = pd.concat([self.df, dummies.astype('int64')], axis=1)
        self.df = self.df.drop(columns=["Modèle"],axis=1)
        return self.df
    
    def encode_marque(self,df):
        self.df = df
        dummies = pd.get_dummies(self.df["Marque"], prefix="Marque")
        self.df = pd.concat([self.df, dummies.astype('int64')], axis=1)
        self.df = self.df.drop(columns=["Marque"],axis=1)
        return self.df
    
    def encode_kilometrage(self,df):
        self.df = df
        def process_kilometrage(value):
            if pd.isna(value):
                return np.nan
                
            str_value = str(value)
            
            if 'Plus de' in str_value or 'plus de' in str_value:
                digits = ''.join(c for c in str_value if c.isdigit())
                if digits:  
                    return int(float(digits) * 1.1)  
                return np.nan
                
            if '-' in str_value:
                parts = str_value.split('-')
                val1 = ''.join(c for c in parts[0] if c.isdigit())
                val2 = ''.join(c for c in parts[1] if c.isdigit())
                
                if val1 and val2:
                    return int((int(val1) + int(val2)) / 2)
                elif val1:
                    return int(val1)
                elif val2:
                    return int(val2)
                return np.nan
                
            digits = ''.join(c for c in str_value if c.isdigit())
            return int(digits) if digits else np.nan
        
        self.df['Kilométrage'] = self.df['Kilométrage'].map(process_kilometrage)
        
        return self.df
    

    def encode_etat(self, df: pd.DataFrame):
        encodings = {
            'Neuf': 0,          
            'Excellent': 1,      
            'Très bon': 2,       
            'Bon': 3,            
            'Correct': 4,        
            'Pour Pièces': 5,    
            'Endommagé': 6,      
            np.nan: np.nan       
        }
        df['État'] = df['État'].map(encodings)
        
        return df   
    
    def standrize_etat(self, df):
        if df['État'].isna().any():
            raise ValueError("Column 'État' still has missing values. Impute before scaling.")

        scaler = MinMaxScaler()
        self.standrizers["etat scaler"] = scaler
        df['État'] = scaler.fit_transform(df[['État']])

        return df
  
    
    def encode_boite_de_vitesses(self,df: pd.DataFrame):
        df['Boite de vitesses'] = df['Boite de vitesses'].map(lambda x:  0 if x == "Automatique" else (1 if x == "Manuelle" else np.nan))
        return df
    
    def encode_Type_de_carburant(self,df: pd.DataFrame):
        dummies =  pd.get_dummies(df["Type de carburant"], prefix="Carburant").astype("Int64")
        df = pd.concat([df,dummies], axis=1)
        return df.drop(columns="Type de carburant")
    
    def encode_pm(self,df: pd.DataFrame):
        df['Première main'] = df['Première main'].map(lambda x:  0 if x == "Non" else (1 if x == "Oui" else np.nan))
        return df
    
    def encode_puissance_fiscale(self, df):
        df = df.copy()
        df['Puissance fiscale'] = df['Puissance fiscale'].replace('Plus de 41 CV', '42 CV')
        
        def to_int(value):
            try:
                return int(str(value).replace(' CV', '').strip())
            except:
                return np.nan

        df['Puissance fiscale'] = df['Puissance fiscale'].map(to_int)

        return df

    def encode_origine(self, df):
        dummies = pd.get_dummies(df['Origine'], prefix='Origine')
        df = pd.concat([df, dummies.astype('int64')], axis=1)
        return df.drop(columns=['Origine'])

