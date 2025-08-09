import joblib

preparator = joblib.load('preparation_instance.pkl')
print(preparator.get_clean_data())
