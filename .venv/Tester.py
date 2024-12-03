"""
#Variables
my_string_variable = 'Esto es un string'
print(my_string_variable)


my_int_variable = 13
print(my_int_variable)


my_bool_variable = True
print(my_bool_variable)


#Cast
print(type(str(my_int_variable)))


#Concatenación de Strings
print(my_string_variable, str(my_int_variable), my_bool_variable)


#Funcion len()
print('Longitud de my_string_variable:', len(my_string_variable))


#Varias variables en una sola linea
name, surname, alias, age = 'Jorge', 'Contreras', 'delta', 20
print('Me llamo:', name, surname,'\nMi alias es:',  alias,'\nY tengo', age, 'años.')


#Forzar tipado de variables
address: str = "Mi dirección"
print(type(address))
print(address)


#Operadores
print(10//3) #división aproximada a enteros
print(10/3)
print(2**3) #exponencial

print('Hola' + 'Python') #no pone espacios entre cadenas por defecto ni se pueden mezclar tipos de datos
print('Hola', 'Python', 5) #pone espacios entre las cadenas y se pueden mezclarse tipos de datos


#Listas
my_list = list()
my_other_list = []

print(len(my_list))

my_list = [12, 20, 62, 30, 30, 17]

print(my_list)
print(len(my_list))

my_other_list = [23, 1.80, 'Jorge', 'delta']
print(my_other_list)
"""


import pandas as pd

list = [1, 1, 2, 3, 4, 5, 6]


def calculate_c1(solution, *args, **kwargs):
    # dataset = kwargs['dataset']
    # Calcula la cantidad de asignaturas que se imparten en mismas franjas horarias
    cont = 0
    # Convertir a una Serie de Pandas
    series = pd.Series(arr)

    # Contar ocurrencias
    counts = series.value_counts()

    for i in counts:
        if i > 0:
            cont += i - 1

    return cont

print(calculate_c1(list))























