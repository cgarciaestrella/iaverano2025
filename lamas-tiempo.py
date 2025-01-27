# Redes neuronales recurrentes
# Se usa cuando la medida principal es el tiempo

""" Red neuronal recurrente que prediga los datos
    meteorológicos de la ciudad de Lamas"""
    
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

archivo = '/home/cris/Descargas/lamas-temp.txt'
columns = ['Year', 'Month', 'Day', 'Precipitación', 'MaxTemp', 'MinTemp']
data = pd.read_csv(archivo, delim_whitespace=True, names=columns, na_values=-99.9)
data.dropna(inplace=True)

# Crear una fecha
data['Fecha'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data.sort_values('Fecha', inplace=True)

# Seleccionar las columnas para la predicción
data = data[['Precipitación', 'MaxTemp', 'MinTemp']]

# Normalizar los datos
escalado = MinMaxScaler()
data_escalada = escalado.fit_transform(data)

# Función para entrada datos
def crear_secuencia(data, secuencia):
    x, y = [], []
    for i in range(len(data) - secuencia):
        x.append(data[i:i + secuencia])
        y.append(data[i + secuencia])
    return np.array(x), np.array(y)

secuencia = 30
x, y = crear_secuencia(data_escalada, secuencia)

x_train, x_test, y_train, y_test = train_test_split(x,y, 
                                     test_size=0.05, random_state=42)

# Crear el modelo
modelo = Sequential([
    LSTM(50, activation='tanh', return_sequences=True,input_shape=(secuencia,3)),
    LSTM(50, activation='tanh'),
    Dense(3)
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
modelo.fit(x_train, y_train, epochs=50, batch_size=32,
           validation_data=(x_test, y_test), verbose=1)

# Generar predicciones
inicio = pd.Timestamp('2014-08-01')
fin = pd.Timestamp('2014-08-31')
dias = (fin - inicio).days + 1

ultimos_dias = data_escalada[-secuencia:]
predicciones_escadaladas = []

for _ in range(dias):
    siguiente_prediccion = modelo.predict(ultimos_dias[np.newaxis, :, :])[0]
    predicciones_escadaladas.append(siguiente_prediccion)
    ultimos_dias = np.vstack([ultimos_dias[1:], siguiente_prediccion])
    
predicciones = escalado.inverse_transform(predicciones_escadaladas)

# Dataframe para las predicciones
fechas_predecidas = pd.date_range(start=inicio, end=fin)
predicciones_df = pd.DataFrame(
    predicciones, columns=['Precipitación', 'MaxTemp', 'MinTemp'],
        index=fechas_predecidas
)

# Mostrar predicciones
print("Predicciones para agosto 2014:")
print(predicciones_df)







    





















    


















 