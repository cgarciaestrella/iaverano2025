import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

data_dir = "/home/cris/Descargas/fracturas/"
clean_data_dir = "/home/cris/Descargas/fracturas_clean/"

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"No se encontró el directorio {data_dir}. Verifica la ruta.")

img_size = (224, 224)
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
                    with Image.open(source_path) as img:
                        img = img.convert("RGB")
                        img.save(target_path, "JPEG", quality=95)
                except (IOError, SyntaxError):
                    print(f"❌ Eliminando imagen corrupta: {source_path}")
                    os.remove(source_path)

clean_and_fix_images(data_dir, clean_data_dir)

def load_dataset(directory, subset):
    subset_path = os.path.join(directory, subset)  
    return keras.preprocessing.image_dataset_from_directory(
        subset_path,
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary"
    )

train_dataset = load_dataset(clean_data_dir, "train")
validation_dataset = load_dataset(clean_data_dir, "val")

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomContrast(0.3),
])

base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  

model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  
])

lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(-epoch / 20))

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

epochs = 30
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping, lr_schedule]
)

plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

base_model.trainable = True
for layer in base_model.layers[:-30]:  
    layer.trainable = False  

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    callbacks=[early_stopping]
)

def predict_fracture(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = "Fracturado" if prediction[0][0] > 0.5 else "No fracturado"
    confidence = max(prediction[0][0], 1 - prediction[0][0]) * 100

    plt.imshow(img)
    plt.title(f"Predicción: {predicted_class} \nConfianza: {confidence:.2f}%")
    plt.axis("off")
    plt.show()
    
    return predicted_class, confidence

def predict_fracture_directory(directory_path):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"No se encontró el directorio {directory_path}. Verifica la ruta.")

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
            try:
                with Image.open(file_path) as img:
                    img.verify()
                predict_fracture(file_path)
            except Exception as e:
                print(f"Imagen dañada, no se pudo procesar: {file_path}")

predict_fracture_directory("/home/cris/Descargas/rx")
