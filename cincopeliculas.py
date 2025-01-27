import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

movies = pd.read_csv('/home/cris/Descargas/movie.csv')
ratings = pd.read_csv('/home/cris/Descargas/rating.csv')

# Calcular el número de vistas por película
movie_views = ratings['movieId'].value_counts().reset_index()
movie_views.columns = ['movieId', 'views']

# Combinar
movies = movies.merge(movie_views, on='movieId')

# Preprocesar los datos
movies['genres'] = movies['genres'].apply(lambda x: x.split('|')[0])
label_encoder = LabelEncoder()
movies['genre_encoded'] = label_encoder.fit_transform(movies['genres'])

# Escalado
scaler = MinMaxScaler()
movies['views_scaled'] = scaler.fit_transform(movies[['views']])

x = movies[['movieId', 'genre_encoded']].values
y = movies['views_scaled'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, 
                                                    random_state=42)
# Construir la red neuronal
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu')) # Capa 1
model.add(Dense(32, activation='relu')) # Capa 2
model.add(Dense(16, activation='relu')) # Capa 3
model.add(Dense(1, activation='linear')) # Salida
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Entrenar el modelo
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                    epochs=50, batch_size=16)
# Evaluación
loss, mae = model.evaluate(x_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}, MAE: {mae}")

# Predicción
movies['predicted_views'] = model.predict(x)
top_movies = movies.sort_values(by='predicted_views', ascending=False).head(5)

print("Top 5 películas más vistas: ")
print(top_movies[['title', 'predicted_views']])

plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida Entrenamiento vs Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdidas (MSE)')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
plt.plot(history.history['mae'], label='MAE de entrenamiento')
plt.plot(history.history['val_mae'], label='MAE de validación')
plt.title('Pérdida Entrenamiento vs Validación (MAE)')
plt.xlabel('Épocas')
plt.ylabel('Error Absoluto Medio')
plt.legend()
plt.show()

predicted_views_scaled = scaler.inverse_transform(movies[['predicted_views']])
movies['predicted_views_actual'] = predicted_views_scaled

heatmap_data = movies[['views', 'predicted_views_actual']]
sns.heatmap(heatmap_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación')
plt.show()





























































