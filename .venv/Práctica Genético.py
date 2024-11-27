dataset = {"n_courses" : 3,
           "n_days" : 3,
           "n_hours_day" : 3,
           "courses" : [("IA", 1), ("ALG", 2), ("BD", 3)]}

import random

def generate_random_array_int(alphabet, length):
    # Genera un array de enteros aleatorios de tamaño length
    # usando el alfabeto dado
    rand_array = [0]*length

    for i in range(length):
      rand_array[i] = random.choice(alphabet)

    return rand_array

def generate_initial_population_timetabling(pop_size, *args, **kwargs):
    # Dataset con la misma estructura que el ejemplo
    # Obtener el alfabeto y la longitud a partir del dataset
    # Genera una población inicial de tamaño pop_size
    length = dataset['n_days'] * dataset['n_hours_day']
    alphabet = list(range(length))
    initial_population = []

    candidate_length = 0
    for i in dataset['courses']:
      candidate_length = candidate_length + i[1]

    for i in range(pop_size):
      candidate = generate_random_array_int(alphabet, candidate_length)
      initial_population.append(candidate)

    return initial_population

generate_random_array_int(list(range(9)), 6)
print(generate_initial_population_timetabling(6))

################################# NO TOCAR #################################
#                                                                          #
def print_timetabling_solution(solution, dataset):
    # Imprime una solución de timetabling
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Crea una matriz de n_days x n_hours_day
    timetable = [[[] for _ in range(n_hours_day)] for _ in range(n_days)]

    # Llena la matriz con las asignaturas
    i = 0
    max_len = 6 # Longitud del título Día XX
    for course in courses:
        for _ in range(course[1]):
            day = solution[i] // n_hours_day
            hour = solution[i] % n_hours_day
            timetable[day][hour].append(course[0])
            i += 1
            # Calcula la longitud máxima del nombre de las asignaturas
            # en una misma franja horaria
            max_len = max(max_len, len('/'.join(timetable[day][hour])))

    # Imprime la matriz con formato de tabla markdown
    print('|         |', end='')
    for i in range(n_days):
        print(f' Día {i+1:<2}{" "*(max_len-6)} |', end='')
    print()
    print('|---------|', end='')
    for i in range(n_days):
        print(f'-{"-"*max_len}-|', end='')
    print()
    for j in range(n_hours_day):
        print(f'| Hora {j+1:<2} |', end='')
        for i in range(n_days):
            s = '/'.join(timetable[i][j])
            print(f' {s}{" "*(max_len-len(s))}', end=' |')
        print()
#                                                                          #
################################# NO TOCAR #################################

# Ejemplo de uso de la función generar individuo con el dataset de ejemplo
candidate = generate_random_array_int(list(range(9)), 6)
print(candidate)
print_timetabling_solution(candidate, dataset)

"""### Función de fitness"""

def create_timetable(candidate):
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Crea una matriz de n_days x n_hours_day
    timetable = [[[] for _ in range(n_hours_day)] for _ in range(n_days)]

    # Llena la matriz con las asignaturas
    i = 0
    max_len = 6 # Longitud del título Día XX
    for course in courses:
        for _ in range(course[1]):
            day = candidate[i] // n_hours_day
            hour = candidate[i] % n_hours_day
            timetable[day][hour].append(course[0])
            i += 1
            # Calcula la longitud máxima del nombre de las asignaturas
            # en una misma franja horaria
            max_len = max(max_len, len('/'.join(timetable[day][hour])))
    return timetable

def calculate_c1(solution, *args, **kwargs):
    #dataset = kwargs['dataset']
    # Calcula la cantidad de asignaturas que se imparten en mismas franjas horarias
    error_counter = 0
    for i in range(solution):
      for j in range(solution[0]):
        if solution[i][j].contains("/"):
          error_counter += 1

    return error_counter

def calculate_c2(solution, *args, **kwargs):
    #dataset = kwargs['dataset']
    # Calcula la cantidad de horas por encima de 2 que se imparten
    # de una misma asignatura en un mismo día
    i = 0
    error_counter = 0
    subjects = []
    for i in range(solution):
      for j in range(solution):
        solution[j][i]

    return None

def calculate_p1(solution, *args, **kwargs):
    #dataset = kwargs['dataset']
    # Calcula el número de huecos vacíos entre asignaturas
    return None

def calculate_p2(solution, *args, **kwargs):
    #dataset = kwargs['dataset']
    # Calcula el número de días utilizados en los horarios
    return None

def calculate_p3(solution, *args, **kwargs):
    #dataset = kwargs['dataset']
    # Calcula el número de asignaturas con horas NO consecutivas en un mismo día
    return None

def fitness_timetabling(solution, *args, **kwargs):
    # Calcula el fitness de una solución de timetabling siguiendo la fórmula del enunciado
    if calculate_c1(solution) > 0 or calculate_c2(solution) > 0:
      return 0
    else:
      return 1 / (1 + calculate_p1(solution) + calculate_p2(solution) + calculate_p3(solution))


# Pistas:
# - Una función que devuelva la tabla de horarios de una solución
# - Una función que devuelva la cantidad de horas por día de cada asignatura
# - A través de args y kwargs se pueden pasar argumentos adicionales que vayamos a necesitar

fitness_timetabling(candidate, dataset=dataset) # Devuelve la fitness del candidato de ejemplo

"""## Operadores genéticos

### Selección por torneo
"""

def select_best(fitness_of_candidates, *args, **kwargs):
  #Calcula los mejores candidatos y los mete en una lista
    i = 0
    best_candidates = []
    if len(fitness_of_candidates) % 2 == 0:
      while i < (fitness_of_candidates - 1):
        if max(fitness_of_candidates[i[1]], fitness_of_candidates[i + 1[1]]) == fitness_of_candidates[i]:
          best_candidates.append(fitness_of_candidates[i])
        else:
          best_candidates.append(fitness_of_candidates[i + 1])
        i += 2
    return best_candidates


def tournament_selection(population, fitness, number_parents, *args, **kwargs):
    t = []  # Tamaño del torneo
    # Selecciona number_parents individuos de la población mediante selección por torneo

    #Genera un numero random
    selector = round(random.uniform(0.0, 1.0), 2)

    #Genera una población inicial(mas adelante se hará con population)
    initial_population = generate_initial_population_timetabling(6)

    #Genera un índice para elegir los predecesores
    index = int(selector * len(initial_population)) - 1

    #Elige los predecesores con el number_parents
    i = 0
    while i < number_parents:
      t.append(initial_population[index])
      i += 1

    #Calcula la fitness de los elegidos para el torneo
    fitness_of_candidates = []
    for i in t:
      fitness_of_candidates.append(((i, fitness_timetabling(i))))

    #Elige de dos en dos el mejor por la fitness
    the_best_ones =[].append(select_best(fitness_of_candidates))

    return the_best_ones


tournament_selection(generate_initial_population_timetabling, fitness_timetabling, 4)
# Pista:
# - Crear una función auxiliar que genere un padre a partir de una selección por torneo
# - Recuerda usar la misma librería de números aleatorios que en el resto del código

"""### Cruce de un punto"""

def one_point_crossover(parent1, parent2, p_cross, *args, **kwargs):
    # Realiza el cruce de dos padres con una probabilidad p_cross
    return None, None

"""### Mutación uniforme"""

def uniform_mutation(chromosome, p_mut, *args, **kwargs):
    dataset = kwargs['dataset'] # Dataset con la misma estructura que el ejemplo
    # Realiza la mutación gen a gen con una probabilidad p_mut
    # Obtener el alfabeto del dataset para aplicar la mutación
    return None

"""### Selección ambiental (reemplazo generacional)"""

def generational_replacement(population, fitness, offspring, fitness_offspring, *args, **kwargs):
    # Realiza la sustitución generacional de la población
    # Debe devolver tanto la nueva población como el fitness de la misma
    return None, None

"""## Algoritmo genético

### Condición de parada (número de generaciones)
"""

def generation_stop(generation, fitness, *args, **kwargs):
    max_gen=kwargs['max_gen']
    # Comprueba si se cumple el criterio de parada (máximo número de generaciones)
    return None

"""### Algoritmo genético"""

def genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                      selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Aplica un algoritmo genético a un problema de maximización
    population = None # Crea la población de individuos de tamaño pop_size
    fitness = None # Contiene la evaluación de la población
    best_fitness = [] # Guarda el mejor fitness de cada generación
    mean_fitness = [] # Guarda el fitness medio de cada generación
    generation = 0 # Contador de generaciones

    # 1 - Inicializa la población con la función generate_population
    # 2 - Evalúa la población con la función fitness_function
    # 3 - Mientras no se cumpla el criterio de parada stopping_criteria
    # 4 - Selección de padres con la función selection
    # 5 - Cruce de padres mediante la función crossover con probabilidad p_cross
    # 6 - Mutación de los descendientes con la función mutation con probabilidad p_mut
    # 7 - Evaluación de los descendientes
    # 8 - Generación de la nueva población con la función environmental_selection

    return population, fitness, generation, best_fitness, mean_fitness