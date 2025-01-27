# Programa para ordenar una matriz
def ingresar_matriz(m,n):
    matriz = []
    print(f"Ingrese los elementos de la matriz de {m}x{n}:")
    for i in range(m):
        fila = []
        for j in range(n):
            elemento = int(input(f"Elemento [{i+1},{j+1}]: "))
            fila.append(elemento)
        matriz.append(fila)
    return matriz

def mostrar_matriz(matriz, titulo="Oño"):
    print(titulo)
    for fila in matriz:
        print(" ".join(map(str, fila)))

m = int(input("Ingrese el número de filas (m): "))
n = int(input("Ingrese el número de columnas (n): "))
matriz = ingresar_matriz(m, n) 
mostrar_matriz(matriz, "Matriz original")
       
    