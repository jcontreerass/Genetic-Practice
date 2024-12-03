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



list = [0, 3, 8, 5, 3, 4]


dataset = {"n_courses" : 3,
           "n_days" : 3,
           "n_hours_day" : 3,
           "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}

def calculate_c2(solution, *args, **kwargs):
    # dataset = kwargs['dataset']
    # Calcula la cantidad de horas por encima de 2 que se imparten
    # de una misma asignatura en un mismo día
    # Dividir entre 3 y ver que numeros son iguales

    aux = [0] * len(solution)
    counter = 0
    courses = dataset['courses']

    for i in range(len(solution)):
        aux[i] = solution[i] // dataset['n_days']

    i = 0
    n_days = dataset['n_days'] - 1

    for course in courses:
        array = [0] * 3
        for _ in range(course[1]):
            array[aux[i]] += 1
            i += 1
            print(array)
            counter = sum(array[i] - n_days if array[i] > 2 else 0 for i in range(_))

    return counter


print(calculate_c2(list))























