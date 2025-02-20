from tensorflow.keras.models import load_model
model = load_model("modelo_cacao.h5")

import numpy as np
from tensorflow.keras.preprocessing import image

img_path = "/home/cris/Descargas/170152_web.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)/255
img_array = np.expand_dims(img_array, axis=0)

class_labels = ["Pudrición parda", "Frutos sanos", "Bicolor"]

img_path = "/home/cris/Descargas/Cocoa-tree-980x380-1-800x380.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0  
img_array = np.expand_dims(img_array, axis=0)  

predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)
predicted_class = class_labels[predicted_class_index]

print(f"Predicción: {predicted_class}")
print("Probabilidades por clase:")
for i, label in enumerate(class_labels):
    print(f"{label}: {predictions[0][i]*100:.2f}%")

