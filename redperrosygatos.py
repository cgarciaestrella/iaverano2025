# g h G H {} []

# Redes neuronales convolucionales

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

data_dir="/home/cris/Descargas/cats_and_dogs_filtered/"

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"No se encontró el directorio {data_dir}. Verifica la ruta")

dataset_train = os.path.join(data_dir, 'train')
dataset_validation = os.path.join(data_dir, 'validation') 
dataset_test = os.path.join(data_dir, 'test')

if not os.path.exists(dataset_train) or not os.path.exists(dataset_validation):
    raise FileNotFoundError("No se encontró carpetas de entrenamiento o validación")

# Procesamiento de imágenes
img_size = (180, 180)
batch_size = 32

dataset_train = keras.preprocessing.image_dataset_from_directory(
    dataset_train,
    image_size = img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=42
)
dataset_validation = keras.preprocessing.image_dataset_from_directory(
    dataset_validation,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=42
)
# Cargar dataset de prueba
if os.path.exists(dataset_test):
    dataset_test = keras.preprocessing.image_dataset_from_directory(
        dataset_test,
        image_size = img_size,
        batch_size=batch_size
        )
else:
    dataset_test = None

class_names = dataset_train.class_names
plt.figure(figsize=(10,10))
for images, labels in dataset_train.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Creación de modelo
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(180,180,3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenamiento
epochs = 10
history = model.fit(dataset_train, validation_data = dataset_validation, epochs=epochs)

#Mostrar precisión
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Evaluación
if dataset_test:
    test_loss, test_acc = model.evaluate(dataset_test)
    print(f"Precisión en dataset en prueba {test_acc * 100:.2f}%")




















            







    
    















    
    
    
    
    
    