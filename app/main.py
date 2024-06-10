import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Crear un dataset de muestra
data = {
    'Flujo_pasajeros': [100, 200, 300, 400, 500],
    'Tiempo_espera': [10, 15, 20, 25, 30],
    'Distancia_recorrida': [5, 10, 15, 20, 25]
}

df = pd.DataFrame(data)

# Dividir el dataset en conjunto de entrenamiento y prueba
X = df[['Flujo_pasajeros', 'Distancia_recorrida']]
y = df['Tiempo_espera']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones
predictions = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, predictions)
print('Error cuadrático medio:', mse)
