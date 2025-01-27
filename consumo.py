#Consumo de luz basado en temperatura
# Regresión lineal simple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
temperatura = np.random.uniform(10,35, 100)
consumo = 50 * 2.5 * temperatura + np.random.normal(0,5,100)

data = pd.DataFrame({'Temperatura': temperatura, 'Consumo': consumo})

x = data[['Temperatura']]
y = data['Consumo']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#Crear el modelo
modelo = LinearRegression()
modelo.fit(x_train, y_train)

#Predicción
y_pred = modelo.predict(x_test)

#Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred) 

plt.figure(figsize=(10,6))
plt.scatter(x_test, y_test, color="blue", label='Datos reales' )
plt.plot(x_test, y_pred, color="red", label="Línea de regresión")
plt.xlabel('Temperatura °C')
plt.ylabel('Consumo Kw')
plt.title('Consumo vs Temperatura')
plt.legend()
plt.grid()
plt.show()








