import numpy as np
matriz = np.random.randint(1,100, (5,5))
print("Matriz original")
print(matriz)

#Operaciones
print("\nSuma de todos los elementos: ",np.sum(matriz))
print("Media de la matriz: ",np.mean(matriz))
print("Máximo de cada fila: ",np.max(matriz, axis=1))
print("Mínimo de cada fila: ",np.min(matriz, axis=1))

#Ordenar matriz
matriz_ordenada = np.sort(matriz, axis=1)
print("\nMatriz Ordenada (por filas): ")
print(matriz_ordenada)