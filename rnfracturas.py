# g H h G [] {}

# Red neuronal convolucional para determinar fracturas

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import image
   
data_dir = '/home/cris/Descargas/fracturas/'
clean_data_dir = '/home/cris/Descargas/fracturas_clean/'

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"No se encontró el directorio {data_dir}. Verificar")

img_size=(224, 224)
batch_size = 32

def clean_and_fix_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_path = os.path.join(root, file)
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                try:
                    with image.open(source_path) as img:
                        img = img.convert('RGB')
                        img.save(target_path,'JPEG', quality=95)
                except (IOError, SyntaxError):
                    print(f"Eliminando imagen corrupta: {source_path}")
                    os.remove(source_path)
clean_and_fix_images(data_dir, clean_data_dir)                    

def load_dataset(directory):
    try:
        dataset = image_dataset_from_directory(
            directory,
            image_size = img_size,
            batch_size = batch_size,
            label_model = 'binary',
            shuffle = True,
            seed = 42
        )
        return dataset
    except Exception as e:
        print(f"Error al cargar las imagenes en {directory}: {e}")
        return None
train_dataset = load_dataset(os.path.join(clean_data_dir, "train"))
test_dataset = load_dataset(os.path.join(clean_data_dir, "test"))
validation_dataset = load_dataset(os.path.join(clean_data_dir, "val"))   

if not train_dataset or not validation_dataset:
    raise RuntimeError("Error con los datasets, verifica las imágenes")

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)            
])    
base_model = keras.applications.MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,
    weights="imagenet"
)
base_model.trainable = False
model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAvaragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')    
])
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
epochs = 5
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()
def predict_fracture(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = "Fracturado" if prediction[0][0]>0.5 else "No fracturado"
    confidence = prediction[0][0]*100 if predicted_class == "Fracturado" else (1-prediction[0][0]*100)
    
    plt.imshow(img)
    plt.title(f"Predicción: {predicted_class} \nConfianza: {confidence:.2f}%")
    plt.axis("off")
    plt.show()
    return predicted_class, confidence     
    
def predict_fracture_directory(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"No se encontró directorio {directory_path}")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                with image.open(file_path) as img:
                    img.verify()
                    predict_fracture(file_path)
            except Exception as e:
                print(f"No se pudo procesar la imagen {file_path}")







































 
        
        
        
        
        
        
        
        
        
    










                

















