# Regresión Logística - Determinar si una persona tiene diabetes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

np.random.seed(42)
n_casos = 100

# Características de las variables
glucosa = np.random.uniform(70, 200, n_casos)
presion = np.random.uniform(60, 140, n_casos)
imc = np.random.uniform(18, 40, n_casos)
edad = np.random.randint(20, 80, n_casos)

# Regla
diabetes = ((glucosa > 120) & (imc >25) & (edad>40)).astype(int)

data = pd.DataFrame({
    'glucosa': glucosa,
    'presion': presion,
    'imc': imc,
    'edad': edad,
    'diabetes': diabetes
    })

# Visualización inicial
sns.pairplot(data, hue='diabetes', diag_kind='kde', palette='Set1')
plt.suptitle('Visualización de datos', y=1.02)
plt.show()

x = data[['glucosa', 'presion', 'imc', 'edad']]
y = data['diabetes']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

# Normalización de datos
scaler = StandardScaler()
x_train_escalado = scaler.fit_transform(x_train)
x_test_escalado = scaler.transform(x_test)

modelo = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
modelo.fit(x_train_escalado, y_train)

# Evaluación

y_pred = modelo.predict(x_test_escalado)
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))

# Matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=['Sin diabetes',
            'Con diabetes'], yticklabels=['Sin diabetes', 'Con diabetes'])
plt.title('Matriz de confusión')
plt.ylabel('Clase verdadera')
plt.xlabel('Clase predicha')
plt.show()

# Precision del modelo
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision:.2f}")

# Predicción
# Nuevos datos con valores (glucosa, presion, imc, edad)
# Regla de diabetes = ((glucosa > 120) & (imc >25) & (edad>40))
pacientes = np.array([
    [150, 85, 30, 50], # Sí tiene
    [95, 70, 22, 25], # No tiene
    [130, 75, 28, 45], # Tiene un poquito
    [110, 65, 23, 30], # No tiene
    [180, 90, 35, 60], # Sí tiene
    ])
pacientes_escalados = scaler.transform(pacientes)
prediccion = modelo.predict(pacientes_escalados)

print("\nResultados para nuevos pacientes:\n")
for i, pred in enumerate(prediccion):
    estado = "Diabético" if pred==1 else "No diabético"
    print(f"Paciente {i+1}: {pacientes[i]}=>{estado}")























