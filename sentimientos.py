# Predicción de sentimientos (X - Ex Twitter)
# Red neuronal recurrente
# g h G H {}  []

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

positivos = '/home/cris/Descargas/tweets_pos_clean.txt'
negativos = '/home/cris/Descargas/tweets_neg_clean.txt'
prueba = '/home/cris/Descargas/tweets_clean.txt'

with open(positivos, 'r', encoding='utf-8') as f:
    pos_tweets = f.readlines()
with open(negativos, 'r', encoding='utf-8') as f:
    neg_tweets = f.readlines() 
with open(prueba, 'r', encoding='utf-8') as f:
    test_tweets = f.readlines()

pos_label = [1] * len(pos_tweets)
neg_label = [0] * len(neg_tweets)

tweets = pos_tweets + neg_tweets
labels = pos_label + neg_label

# Creamos el dataframe
data = pd.DataFrame({'tweet': tweets, 'label': labels})

# Juntamos los datos
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

max_palabras = 10000
max_tamaño = 100

tokenizer = Tokenizer(num_words = max_palabras)
tokenizer.fit_on_texts(data['tweet'])
sequences = tokenizer.texts_to_sequences(data['tweet'])
x_data = pad_sequences(sequences, maxlen=max_tamaño)
y_data = np.array(data['label'])
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, 
                                                    random_state=42)
modelo = Sequential([
    Embedding(max_palabras, 32, input_length=max_tamaño),
    SimpleRNN(32, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Compilar
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar
modelo.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluación
loss, accuracy = modelo.evaluate(x_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# Predicción de sentimientos
test_sequences = tokenizer.texts_to_sequences(test_tweets)
test_padded = pad_sequences(test_sequences, maxlen=max_tamaño)
predicciones = modelo.predict(test_padded)

# Mostrar resultados
print("Resultados de análisis de sentimientos:")
for tweet, sentimientos in zip(test_tweets, predicciones):
    sentimientos_label = 'Positivo' if sentimientos>0.5 else 'Negativo'
    print(f"Tweet: {tweet.strip()}")
    print(f"Sentimiento Predicho: {sentimientos_label} (Probabilidad: {sentimientos[0]:.2f})")
    print("-" * 50)













 










    