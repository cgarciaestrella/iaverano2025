# Análisis estadístico con Pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Para mapas de calor
import pandas as pd

np.random.seed(42)
# Diccionario de datos
data = { 
        'Edad': np.random.randint(18,70,100),
        'Ingresos': np.random.normal(50000, 15000, 100).astype(int),
        'Categoría' : np.random.choice(['A','B','C'],100)
}
df = pd.DataFrame(data)
print("Primeras filas de DataFrame")
print(df.head())
print(df.describe())

plt.figure(figsize=(10,6))
plt.subplot(1,2,2)
sns.boxplot(x="Categoría",y='Ingresos',data=df, palette='Set1')
plt.title("Ingresos por categoría")
plt.show()
 

