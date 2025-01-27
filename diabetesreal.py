import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar los archivos
datos = '/home/cris/Descargas/datadiabetes.csv'
prediccion = '/home/cris/Descargas/dataprediccion.csv'

data = pd.read_csv(datos)
print("Datos reales:")
print(data.head(10))

# Visualización inicial de datos
sns.pairplot(data, hue='diabetes', diag_kind='kde', palette='Set1')
plt.suptitle('Visualización de datos', y=1.02)
plt.show()

# Separación de datos
x = data.drop(columns=['diabetes'])
y = data['diabetes']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.05, random_state=42)
# Normalizar los datos
escalado = StandardScaler()
x_train_escalado = escalado.fit_transform(x_train)
x_test_escalado = escalado.transform(x_test)

#Entrenar
perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(x_train_escalado, y_train)
# Evaluación
y_pred = perceptron.predict(x_test_escalado)

print('\nReporte de Clasificación\n')
print(classification_report(y_test, y_pred))

# Matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues',
            xticklabels=['SinDiabetes', 'ConDiabetes'],
            yticklabels=['SinDiabetes', 'ConDiabetes'])
plt.title('Matriz')
plt.ylabel('Clase verdadera')
plt.xlabel('Clase predicha')
plt.show()

# Precisión
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision:.2f}")

# Nuevos datos
nuevos_datos = pd.read_csv(prediccion)
print("\n Datos para la predicción\n")
print(nuevos_datos.head(10))

# Normalizar nuevos datos
nuevos_datos_escalado = escalado.transform(nuevos_datos)
predicciones = perceptron.predict(nuevos_datos_escalado)

print("\n Resultados de la predicción \n")
for i, pred in enumerate(predicciones):
    estado = 'ConDiabetes' if pred == 1 else 'SinDiabetes'
    print(f"Datos {i+1}: {nuevos_datos.iloc[i].tolist()} => {estado}")






























    
    