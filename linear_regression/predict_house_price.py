from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Exemplo de dados
dados = {
    "tamanho_m2": [50, 80, 120, 150],
    "quartos": [1, 2, 3, 4],
    "preco": [200000, 320000, 500000, 650000]
}
df = pd.DataFrame(dados)

# Features (X) e alvo (y)
X = df[["tamanho_m2", "quartos"]]
y = df["preco"]

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Previsão
previsao = modelo.predict([[100, 3]])
print(f"Preço estimado da casa: R${previsao[0]:,.2f}")