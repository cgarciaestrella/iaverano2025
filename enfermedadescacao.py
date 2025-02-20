# Predicción de enfermedades de cacao
# g G H h

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

base_dir = "/home/cris/Descargas/cacao_diseases/cacao_photos/"
train_dir = base_dir

datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 30,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    validation_split = 0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training'    
)
val_generator = datagen.flow_from_directory(
    train_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation'    
)
base_model = EfficientNetB0(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224,224,3)
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(3, activation='softmax')

model = Model(inputs=base_model.input, outputs=output_layer(x))
model.compile(
    optimizer = Adam(learning_rate=0.0001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
history = model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = 50,
    steps_per_epoch = len(train_generator),
    validation_steps = len(val_generator)       
)

model.save("modelo_cacao.keras")

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    base_dir,
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical'
)

eval_results = model.evaluate(test_generator)
print(f"Pérdida: {eval_results[0]}, Entrenamiento: {eval_results[1]}")