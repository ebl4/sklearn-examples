import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Exemplo: Pré-processamento completo
def preprocess_data(data):
    # Lidar com valores missing
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data.select_dtypes(include=[np.number]))
    
    # Normalizar dados numéricos
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    return data_scaled

# Dados de exemplo
data = pd.DataFrame({
    'idade': [25, 30, np.nan, 35, 40],
    'salario': [50000, 60000, 70000, np.nan, 90000],
    'cidade': ['SP', 'RJ', 'SP', 'MG', 'RJ']
})

# Pré-processamento
numerical_data = data[['idade', 'salario']]
processed_numerical = preprocess_data(numerical_data)
print("Dados processados:\n", processed_numerical)