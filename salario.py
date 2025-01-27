""" Predcción de salario (y) basado en lo siuiente:
    1. Años de experiencia
    2. Nivel educativo (1: Baciller, 2: Maestría, 3: Doctorado
    3. oras trabajadas
    4. Edad
    5. Calificación de desempeño (0 - 100)
    (x)
    Entre tanto, estamos usando 5 características"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n_personas = 5000

experiencia = np.random.randint(1,30, n_personas)
nivel = np.random.choice([1,2,3], n_personas, p=[0.5, 0.3, 0.2])
horas = np.random.uniform(30, 60, n_personas)
edad = experiencia + np.random.randint(22, 30, n_personas)
evaluacion = np.random.uniform(50, 100, n_personas)
salario = (
    20000 + 1500 * experiencia + 10000*nivel + 400*horas-200 * edad
     +300 * evaluacion + np.random.normal(0, 15000, n_personas)
    )

data = pd.DataFrame({
    'Experiencia': experiencia,
    'Nivel': nivel,
    'Horas': horas,
    'Edad': edad,
    'Evaluacion': evaluacion,
    'Salario': salario
    })

plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Matriz de correlación", fontsize=16)
plt.show()

x = data[['Experiencia', 'Nivel', 'Horas', 'Edad', 'Evaluacion']]
y = data['Salario']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)

# Modelo
modelo = LinearRegression()
modelo.fit(x_train, y_train)
# Predicción
y_pred = modelo.predict(x_test)
# Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Resultados
print('Coeficientes del modelo')
for feature, coef in zip(x.columns, modelo.coef_):
    print(f" {feature}:{coef:.2f}")

print(f"Intercepción:{modelo.intercept_:.2f}")
print(f"Error cuadrático:{mse:.2f}")
print(f"R2:{r2:.2f}")

# Gráficos

plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
sns.barplot(x=modelo.coef_, y=x.columns, palette='Set2')
plt.title('Importancia de las características')
plt.xlabel('Coeficiente')
plt.ylabel('Características')
plt.grid(axis='x')

plt.subplot(2,2,2)
errores = y_test - y_pred 
sns.histplot(errores, kde=True, color='purple', bins=20)
plt.title('Distribución de errores')
plt.xlabel('Error')
plt.ylabel('Frecuencia')

plt.subplot(2,1,2)
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y.min(), y.max()],[y.min(), y.max()], 'r--', lw=2)
plt.title('Predicción vs Realidad')
plt.xlabel('Valores Reales ')
plt.ylabel('Valores de Predicción')
plt.grid()
plt.tight_layout()
plt.show()



    


 
 













    
    