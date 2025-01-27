# La suma de 2 números
"""x = float(input("Ingrese el primer número: "))
y = float(input("Ingrese el segundo número: "))
suma = x + y
print(f"La suma es {suma}")"""

# La suma de los n primeros números naturales

"""x = int(input("Ingrese la cantidad de elementos: "))
contador = 1
suma = 0
while contador <= x:
    print(contador)
    suma = suma + contador
    contador = contador + 1
print("La suma es: ", suma)   """

n = int(input("Ingrese la cantidad de elementos: "))
array = []
print("Ingrese los elementos del arreglo: ")
for i in range(n):
    elemento = str(input(f"Elemento {i + 1}: "))
    array.append(elemento)
ordenado = sorted(array)
print(array)
print(ordenado)













 
    