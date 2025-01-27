# Crear una gráfico de una ecuación cuadrática
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y=-x**3+5*x**2+8

plt.figure(figsize=(8,6))
#plt.plot(x,y,label="y=-x^3+5x^2+8", color="blue", linewidth=2)
plt.scatter(x,y,label="y=-x^3+5x^2+8", color="blue", linewidth=2)

plt.title("Gráfica", fontsize=16)
plt.xlabel("Eje X", fontsize=14)
plt.ylabel("Eje y", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12)
plt.axhline(0,color="red",linewidth=3)
plt.axvline(0,color="red",linewidth=3)
plt.show()




 
