# Visualización interactiva de datos

import numpy as np
import matplotlib.pyplot as plt

tiempo = np.linspace(0, 10, 100)
amplitud = np.sin(tiempo)

plt.figure(figsize=(10,6))
plt.plot(tiempo, amplitud, label="Onda Senoidal", color='blue', linewidth=2)
plt.title("Gráfico de onda", fontsize=16)
plt.xlabel("Tiempo", fontsize=14)
plt.ylabel("Amplitud", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.8)
plt.legend(fontsize=12, loc="lower right")
plt.show()