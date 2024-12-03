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

import numpy as np

dataset = {"n_courses" : 3,
           "n_days" : 3,
           "n_hours_day" : 3,
           "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}

list = [0, 6, 8, 5, 3, 2]


def calculate_p1(solution, *args, **kwargs):
    #dataset = kwargs['dataset']
    # Calcula el número de huecos vacíos entre asignaturas
    counter  = 0
    n_days = dataset['n_days']

    for i in range(len(solution) - 1):
        arr = solution.copy()
        arr = np.sort(arr)
        if (arr[i] // n_days == arr[i + 1] // n_days) and ((arr[i + 1] - arr[i]) == 2):
            counter += 1

    return counter


print(calculate_p1(list))


"""
return 3
def calculate_p1(solution, *args, **kwargs):
    #dataset = kwargs['dataset']
    # Calcula el número de huecos vacíos entre asignaturas
    counter  = 0
    n_days = dataset['n_days']

    print(n_days)

    for i in range(len(solution) - 1):
        arr = solution.copy()
        arr = np.sort(arr)
        print(arr)
        if (arr[i] // n_days == arr[i + 1] // n_days) and ((arr[i + 1] - arr[i]) == 2):
            counter += 1

    return counter
"""


"""
return 3
def calculate_p2(solution, *args, **kwargs):
    # dataset = kwargs['dataset']
    # Calcula el número de días utilizados en los horarios
    # Sumar 1 a todos los numeros del candidato y modulo entre n_hour_day y si es 0 suma 1 dia
    aux = solution.copy()
    dias = 0
    aux = [(val + 1) % 3 for val in aux]

    for i in aux:
        if i == 0:
            dias += 1

    return dias
"""



"""
return 2
def calculate_p3(solution, *args, **kwargs):
    # dataset = kwargs['dataset']
    # Calcula el número de asignaturas con horas NO consecutivas en un mismo día

    counter = 0
    courses = dataset["courses"]
    i = 0

    for course in courses:
        if course[1] >= 2:
            arr = solution[i : i + course[1]].copy()
            arr = np.sort(arr)
            counter += sum(1 if (arr[k + 1] - arr[k]) == 2 else 0 for k in range(len(arr) - 1))
        i += course[1]

    return counter
"""





















