# Perceptrón (Red neuronal de 1 capa)
# Entradas, pesos, Función de Suma, Función de activación (0, 1)

# Predecir la aprobación de una préstamo
# h g {} []
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

np.random.seed(42)
n_casos = 10000

historial = np.random.randint(0,2, n_casos) # 0: mal historial 1: buen historial
ingresos = np.random.uniform(1000, 10000, n_casos)
edad = np.random.randint(18, 70, n_casos)
deudas = np.random.uniform(0, 5000, n_casos)

aprobado = ((historial == 1) & (ingresos > 5000) & (deudas < 2000)).astype(int)

creditos = pd.DataFrame({
    'historial': historial,
    'ingresos': ingresos,
     "edad": edad,
     'deudas': deudas,
     'aprobado': aprobado
})

# Visualización inicial
sns.pairplot(creditos, hue='aprobado', diag_kind='kde', palette='Set1')
plt.suptitle('Visualización de datos', y=1.02)
plt.show()

x = creditos[['historial', 'ingresos', "edad", 'deudas']]
y = creditos['aprobado']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)

# Normalización de datos
from sklearn.preprocessing import StandardScaler
escalado = StandardScaler()
x_train_escalado = escalado.fit_transform(x_train)
x_test_escalado = escalado.transform(x_test)

# Entrenar el perceptrón
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(x_train_escalado, y_train)

#Evaluación
y_pred = perceptron.predict(x_test_escalado)
print("\nReport de Clasificación:\n")
print(classification_report(y_test, y_pred))

# Matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
sns.heatmap(matriz, annot=True, fmt='d',cmap='Blues', xticklabels=['Rechazado','Aprobado'],
            yticklabels=['Rechazado','Aprobado'])
plt.title('Matriz de confusión')
plt.ylabel('Clase verdadera')
plt.xlabel('Clase predicha')
plt.show()

# Precisión del modelo
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision:.2f}")

# Predecir nuevos datos
# historial, ingresos, edad, deudas
nuevos_datos = np.array([
    [1, 6000, 30, 1000],
    [0, 4000, 45, 2500],
    [1, 8000, 29, 1500],
    [0, 2000, 50, 3000],
    [1, 7000, 35, 1200],
    [1, 3000, 40, 4000],
    [0, 1000, 25, 500],
    [1, 9000, 27, 1000],
    [0, 2500, 33, 2000],
    [1, 5000, 38, 1800],
    [0, 1500, 60, 3500],
    [1, 6500, 45, 1000],
    [0, 3500, 55, 4500],
    [1, 4800, 36, 1900],
    [0, 2900, 22, 2200],
    [1, 7500, 29, 900],
    [0, 1800, 60, 4000]
])


nuevos_datos_escalados = escalado.transform(nuevos_datos)
nuevas_predicciones = perceptron.predict(nuevos_datos)

print("\nResultados para nuevos datos:\n")
for i, pred in enumerate(nuevas_predicciones):
    estado = 'Aprobado' if pred == 1 else "Rechazado"
    print(f"Datos {i+1}: {nuevos_datos[i]} => {estado}")
  
    























