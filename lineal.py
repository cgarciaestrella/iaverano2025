import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

x = 2 * np.random.rand(100,1) # Cantidad de equipos
y = 4 + 3*x + np.random.randn(100, 1) # Costo

plt.plot(x,y,"b.")
plt.title("Equipos vs costo")
plt.xlabel("Equipos afectados (1000)")
plt.ylabel("Costo de incidente (1000)")
plt.show()

data = {'n_equpos_afectados': x.flatten(), 'costo' : y.flatten()}
df = pd.DataFrame(data)
print(df.head(10)) #Imprime 10 primeros valores

df['n_equpos_afectados'] = df['n_equpos_afectados']*1000
df['n_equpos_afectados'] = df['n_equpos_afectados'].astype('int')

df['costo'] = df['costo']*10000
df['costo'] = df['costo'].astype('int')
print(df.head(10))

plt.plot(df['n_equpos_afectados'],df['costo'],"b.")
plt.title("Equipos vs costo mejorado")
plt.xlabel("Equipos afectados")
plt.ylabel("Costo de incidente")
plt.show()

lin_reg = LinearRegression() 
lin_reg.fit(df['n_equpos_afectados'].values.reshape(-1,1), df['costo'].values)

x_min_max = np.array([[df['n_equpos_afectados'].min()]])
y_train_pred = lin_reg.predict(x_min_max)

plt.plot(x_min_max, y_train_pred,"r.")
plt.plot(df['n_equpos_afectados'], df['costo'],"g.")
plt.title("Predicción inicial")
plt.xlabel("Equipos afectados")
plt.ylabel("Costo de incidente")
plt.show()

# Predicción

x_nuevo = np.array([[1300]])
costo = lin_reg.predict(x_nuevo)

plt.plot(x_min_max, y_train_pred,"r.")
plt.plot(df['n_equpos_afectados'], df['costo'],"y.")
plt.plot(x_nuevo, costo,"b.")
plt.title("Predicción final")
plt.xlabel("Equipos afectados")
plt.ylabel("Costo de incidente")
plt.show()





