# Con redes neuronales recurrentes también hacemos 
# procesamiento del lenguaje natural
# Entonces vamos a resumir un libro con RNN [] {} 


import numpy as np
import os
import fitz
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
nltk.download('punkt')

# Función para extraer texto
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Preprocesar el texto
def preprocess_text(text):
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
    except:
        print("Tokenización con NLTK")
        sentences = text.split('.')
    return [s.strip() for s in sentences if len(s.strip())>10]

# Calcular la importación semántica (sentido oriinal de las palabras)
def calculate_sentence_importance(sentences, model, tokenizer, maxlen):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen)
    predictions = model.predict(padded_sequences)
    return predictions

# Resumen basado en importancia semántica
def generate_summary(sentences, scores, top_n=5):
    ranked_sentences = [(scores[i], s) for i, s in enumerate(sentences)]
    ranked_sentences.sort(reverse=True, key=lambda x: x[0])
    return " ".join([s[1] for s in ranked_sentences[:top_n]])

# Configurar el modelo
max_words = 10000
maxlen = 100
embedding_dim = 100
top_n = 10

# Entrenar el modelo
def train_model(sentences,labels):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    x = pad_sequences(sequences, maxlen=maxlen)
    y = np.array(labels)
    
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=maxlen),
        Bidirectional(SimpleRNN(64, activation='tanh', return_sequences=False)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
        ])    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x, y, epochs=5, batch_size=32, validation_split=0.2)
    return model, tokenizer

pdf_path = '/home/cris/Descargas/Harry-Potter-y-Las-Reliquias-De-La-Muerte.pdf'
book_text = extract_text_from_pdf(pdf_path)

sentences = preprocess_text(book_text)
labels = np.random.randint(0,2, len(sentences))

print('Entrenando el modelo...')
model, tokenizer = train_model(sentences, labels)

# Importancia de cada frase
print("Calculando importancia de cada frase...")
importance_scores = calculate_sentence_importance(sentences, model, tokenizer, maxlen)

# Generando resumen
print("Generando resumen...")
summary = generate_summary(sentences, importance_scores, top_n=top_n)
print("Resumen generado:")
print(summary)












    
    
    
    


    
        












