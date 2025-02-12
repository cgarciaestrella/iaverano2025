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
    
        
        
        
        
        
        
        
        
        
    










                

















