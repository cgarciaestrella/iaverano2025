# Red neuronal de n capas
# Sistema de recomendación de películas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
n_casos = 10000
data = pd.DataFrame({
    'user_id': np.random.randint(1, 1001, n_casos),
    'movie_id': np.random.randint(1, 501, n_casos),
    'rating': np.random.uniform(1, 5, n_casos),
    'genero': np.random.choice(['Accion', 'Comedia', 'Drama', 'Terror', 'Romance'], n_casos),
    'año': np.random.randint(1980, 2024, n_casos)
    })

filtro = data[(data['genero'] == 'Accion') & (data['año'] >= 2000)]

numero_usuarios = filtro['user_id'].nunique()
numero_peliculas = filtro['movie_id'].nunique()
max_user_id = filtro['user_id'].max() + 1
max_move_id = filtro['movie_id'].max() + 1 

x = filtro[['user_id','movie_id' ]].values
y = filtro['rating'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

dimensiones = 64 # Retropropaación
# Parámetros de entrada de usuarios
user_input = Input(shape=(1,), name='User_Input')
user_dim = Embedding(input_dim=max_user_id, output_dim=dimensiones, name="User_Dim")(user_input)
user_flatten = Flatten()(user_dim)

# Parámetros de entrada de películas
movie_input = Input(shape=(1,), name='Movie_Input')
movie_dim = Embedding(input_dim=max_move_id, output_dim=dimensiones, name="Movie_Dim")(movie_input)
movie_flatten = Flatten()(movie_dim)

# Combinación de valores
combinado = Concatenate()([user_flatten, movie_flatten])

# Confiurar las capas
dense_1 = Dense(256, activation='relu')(combinado)
dropout_1 = Dropout(0.3)(dense_1)
dense_2 = Dense(128, activation='relu')(dropout_1)
dropout_2 = Dropout(0.3)(dense_2)
output = Dense(1, activation='linear')(dropout_2)

# Modelo
modelo = Model(inputs=[user_input, movie_input], outputs = output)
modelo.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Entrenar
entrenar = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = modelo.fit(
    [x_train[:,0], x_train[:,1]],
    y_train,
    validation_data=([x_test[:,0], x_test[:,1]], y_test),
    epochs = 1000000, # El punto de equilibrio del modelo
    batch_size=128,
    callbacks=[entrenar],
    verbose=1    
    )

# Evaluar el modelo
y_pred = modelo.predict([x_test[:,0], x_test[:,1]])
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")

# Visualización del entrenamiento
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='(Entrenamiento)')
plt.plot(history.history['loss'], label='(Validación)')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.show()

# Predicción de recomendaciones
user_id = 10
movie_id = 499
prediccion = modelo.predict([np.array([user_id]), np.array([movie_id])])
print(f"Predicción del usuario {user_id} y la película {movie_id}: {prediccion[0][0]:.2f}")

#Eficiencia del modelo
plt.figure(figsize=(12,6))
plt.scatter(y_test, y_pred, alpha=0.5, label='Predicciones vs Reales')
plt.plot([1,5], [1, 5], color='red', linestyle='--', label='Línea ideal')
plt.xlabel('Calificaciones reales')
plt.ylabel('Calificaciones predichas')
plt.title('Eficiencia del modelo')
plt.show()



























































