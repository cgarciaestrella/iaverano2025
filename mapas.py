import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
x = np.random.normal(0,1,1000)
y = np.random.normal(0,1,1000)

plt.figure(figsize=(10,8))
plt.scatter(x, y, alpha=0.6, edgecolors='w', label="Datos", c='red')
sns.kdeplot(x, cmap="Reds", fill=True, alpha=0.5, levels=30, label="Densidad")

plt.title("Mapa de calor", fontsize=16)
plt.xlabel("Eje x", fontsize=14)
plt.ylabel("Eje y", fontsize=14)
plt.grid(True)
plt.legend(fontsize=12, loc="upper right")
plt.show()
