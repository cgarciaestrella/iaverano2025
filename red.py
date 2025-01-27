# Red Neuronal simple
import numpy as np
from keras.models import Sequential # Recorrer la red
from keras.layers import Dense # Retropropagación
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)
x = np.random.rand(1000, 5)
y = (np.sum(x, axis=1)>2.5).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                      test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


