# Regresión lineal polinómica
# Rendimiento de un motor con la velocidad de un vehículo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
velocidad = np.random.uniform(20,120, 1000)
rendimiento = 25-0.05*(velocidad - 60)**2+np.random.normal(0, 1.5, 1000)

data = pd.DataFrame({'Velocidad': velocidad, 'Rendimiento': rendimiento})
x = data[['Velocidad']]
y = data['Rendimiento']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

poly_features = PolynomialFeatures(degree=2)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)

modelo_poli = LinearRegression()
modelo_poli.fit(x_train_poly, y_train)

# Predicción
y_pred = modelo_poli.predict(x_test_poly)

#Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Visualización
x_line = np.linspace(20,120,300).reshape(-1,1)
x_line_poly = poly_features.transform(x_line)
y_line = modelo_poli.predict(x_line_poly)

plt.figure(figsize=(10,6))
plt.scatter(x, y, color='blue', label='Datos reales')
plt.plot(x_line, y_line, color="red",label='Regresión Polinómica')
plt.xlabel("Velocidad")
plt.ylabel("Rendimiento")
plt.title('Rendimiento vs Velocidad')
plt.legend()
plt.grid()
plt.show()










