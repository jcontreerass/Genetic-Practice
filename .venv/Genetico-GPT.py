import numpy as np
import random


def genetic_algorithm(generate_population, pop_size, fitness_function, stopping_criteria, offspring_size,
                      selection, crossover, p_cross, mutation, p_mut, environmental_selection, *args, **kwargs):
    # Inicialización
    population = generate_population(pop_size, *args, **kwargs)  # Generar población inicial
    fitness = [fitness_function(ind, *args, **kwargs) for ind in population]  # Evaluar fitness inicial
    best_fitness = []
    mean_fitness = []
    generation = 0

    # Bucle evolutivo
    while not stopping_criteria(generation, fitness, *args, **kwargs):
        # Selección de padres
        parents = selection(population, fitness, offspring_size, *args, **kwargs)

        # Generación de descendientes (cruce y mutación)
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]  # En caso de número impar
            child1, child2 = crossover(parent1, parent2, p_cross, *args, **kwargs)
            offspring.append(mutation(child1, p_mut, *args, **kwargs))
            offspring.append(mutation(child2, p_mut, *args, **kwargs))

        # Evaluar descendientes
        fitness_offspring = [fitness_function(ind, *args, **kwargs) for ind in offspring]

        # Reemplazo generacional
        population, fitness = environmental_selection(population, fitness, offspring, fitness_offspring, *args,
                                                      **kwargs)

        # Guardar métricas
        best_fitness.append(max(fitness))
        mean_fitness.append(np.mean(fitness))

        generation += 1

    return population, fitness, generation, best_fitness, mean_fitness


def fitness_function(individual, *args, **kwargs):
    dataset = kwargs['dataset']
    n_days = dataset['n_days']
    n_hours_day = dataset['n_hours_day']
    courses = dataset['courses']

    # Reconstruimos el horario desde el cromosoma
    schedule = [[] for _ in range(n_days * n_hours_day)]
    course_hours = []

    # Obtener el número total de horas de cada curso
    for course in courses:
        course_hours.extend([course[0]] * course[1])  # Ejemplo: IA (1) => ['IA']

    # Asignar las horas del individuo a las celdas del horario
    for hour_idx, slot in enumerate(individual):
        schedule[slot].append(course_hours[hour_idx])

    # Convertir el horario en una matriz por días
    daily_schedule = [schedule[i * n_hours_day:(i + 1) * n_hours_day] for i in range(n_days)]

    # --- Restricciones duras (C1 y C2) ---
    c1 = 0  # Penalización por solapamientos
    c2 = 0  # Penalización por más de 2 horas de una misma asignatura al día

    for day in daily_schedule:
        daily_subjects = []
        for hour in day:
            if len(hour) > 1:
                c1 += len(hour) - 1  # Incrementar por cada conflicto (solapamiento)
            daily_subjects.extend(hour)

        # Contar ocurrencias de cada asignatura en el día
        subject_counts = {subject: daily_subjects.count(subject) for subject in set(daily_subjects)}
        for subject, count in subject_counts.items():
            if count > 2:  # Más de 2 horas de una misma asignatura en el día
                c2 += count - 2

    # Si se violan restricciones duras, el fitness es 0
    if c1 > 0 or c2 > 0:
        return 0

    # --- Restricciones suaves ---
    p1 = 0  # Penalización por huecos vacíos entre clases en un día
    p2 = 0  # Penalización por número de días con clases
    p3 = 0  # Penalización por clases no consecutivas en un día

    active_days = 0
    for day in daily_schedule:
        # Verificar huecos vacíos
        active_hours = [1 if hour else 0 for hour in day]  # 1 si hay clase, 0 si no
        gaps = sum(1 for i in range(len(active_hours) - 1) if active_hours[i] == 0 and active_hours[i + 1] == 1)
        p1 += gaps

        # Contar días con clases
        if any(active_hours):
            active_days += 1

        # Penalizar clases no consecutivas
        for subject in set(course_hours):
            subject_occurrences = [i for i, hour in enumerate(day) if subject in hour]
            if len(subject_occurrences) > 1:
                for i in range(len(subject_occurrences) - 1):
                    if subject_occurrences[i + 1] - subject_occurrences[i] > 1:
                        p3 += 1

    # Penalizar número de días utilizados (queremos un horario compacto)
    p2 = active_days - 1  # Penalizamos usar más días del mínimo requerido

    # --- Función Fitness ---
    fitness = 1 / (1 + p1 + p2 + p3)
    return fitness


"""
OPCION 1
"""


def generate_population(pop_size, *args, **kwargs):
    dataset = kwargs['dataset']
    n_days = dataset["n_days"]
    n_hours_day = dataset["n_hours_day"]
    n_courses = len(dataset["courses"])
    total_hours = sum([course[1] for course in dataset["courses"]])

    # Alfabeto: índices de las franjas horarias disponibles
    alphabet = list(range(n_days * n_hours_day))
    population = []

    for _ in range(pop_size):
        # Generar una solución aleatoria que cumpla con las horas totales
        individual = random.sample(alphabet, total_hours)
        population.append(individual)

    return population


def fitness_population(population, fitness_function, *args, **kwargs):
    return [fitness_function(individual, *args, **kwargs) for individual in population]


def stopping_criteria(generation, fitness, *args, **kwargs):
    max_gen = kwargs['max_gen']
    return generation >= max_gen


def tournament_selection(population, fitness, number_parents, *args, **kwargs):
    tournament_size = kwargs['tournament_size']
    parents = []

    for _ in range(number_parents):
        # Selección aleatoria de participantes del torneo
        tournament_indices = random.sample(range(len(population)), tournament_size)
        # Evaluar sus fitness y elegir el mejor
        best_index = min(tournament_indices, key=lambda idx: fitness[idx])  # Minimización
        parents.append(population[best_index])

    return parents


def one_point_crossover(parent1, parent2, p_cross, *args, **kwargs):
    if random.random() < p_cross:
        point = random.randint(1, len(parent1) - 1)  # Punto de cruce
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        # Sin cruce, los hijos son copias de los padres
        return parent1, parent2


def uniform_mutation(chromosome, p_mut, *args, **kwargs):
    dataset = kwargs['dataset']
    n_days = dataset["n_days"]
    n_hours_day = dataset["n_hours_day"]
    alphabet = list(range(n_days * n_hours_day))  # Posibles franjas horarias

    for i in range(len(chromosome)):
        if random.random() < p_mut:
            chromosome[i] = random.choice(alphabet)  # Reemplaza el gen por uno aleatorio

    return chromosome


def generational_replacement(population, fitness, offspring, fitness_offspring, *args, **kwargs):
    combined_population = population + offspring
    combined_fitness = fitness + fitness_offspring

    # Ordenar por fitness (minimización)
    sorted_indices = sorted(range(len(combined_population)), key=lambda idx: combined_fitness[idx])
    new_population = [combined_population[i] for i in sorted_indices[:len(population)]]
    new_fitness = [combined_fitness[i] for i in sorted_indices[:len(population)]]

    return new_population, new_fitness



#USO COMPLETO DEL ALGORITMO
# Configuración del dataset
dataset = {
    "n_courses": 3,
    "n_days": 3,
    "n_hours_day": 3,
    "courses": [("IA", 1), ("ALG", 2), ("BD", 3)]
}

# Parámetros
pop_size = 100
offspring_size = 50
p_cross = 0.8
p_mut = 0.1
max_gen = 200



population, fitness, generation, best_fitness, mean_fitness = genetic_algorithm(
    generate_population,
    pop_size,
    fitness_population,
    stopping_criteria,
    offspring_size,
    tournament_selection,
    one_point_crossover,
    p_cross,
    uniform_mutation,
    p_mut,
    generational_replacement,
    dataset=dataset,
    tournament_size=5,
    max_gen=max_gen
)

print(f"Mejor fitness alcanzado: {max(best_fitness)}")
print(f"Generaciones ejecutadas: {generation}")




"""
Explicación Adicional
Flujo General del Algoritmo Genético
Inicialización: Se genera una población inicial utilizando generate_population.
Evaluación: Se evalúa la aptitud de cada individuo con fitness_population.
Selección: Se eligen los padres para el cruce mediante tournament_selection.
Cruce: Los padres producen descendientes con one_point_crossover.
Mutación: Los descendientes se mutan con uniform_mutation.
Sustitución: La población actual se reemplaza con los mejores individuos usando generational_replacement.
Parada: El algoritmo se detiene según stopping_criteria.
Puntos a Personalizar
Las restricciones y penalizaciones se implementan dentro de fitness_function.
Las probabilidades de cruce (p_cross) y mutación (p_mut) pueden ajustarse según experimentos.
"""


"""
OPCION 2
"""
""""


def generate_initial_population_timetabling(pop_size, *args, **kwargs):
    dataset = kwargs['dataset']
    n_days = dataset["n_days"]
    n_hours_day = dataset["n_hours_day"]
    n_courses = len(dataset["courses"])

    # Calculamos el total de horas de todas las asignaturas
    total_hours = sum([course[1] for course in dataset["courses"]])

    # Alfabeto: posibles franjas horarias
    alphabet = list(range(n_days * n_hours_day))

    # Genera individuos aleatorios: asigna horas del alfabeto a las asignaturas
    population = [
        generate_random_array_int(alphabet, total_hours) for _ in range(pop_size)
    ]
    return population



def fitness_population(population, fitness_function, *args, **kwargs):
    return [fitness_function(ind, *args, **kwargs) for ind in population]


def generation_stop(generation, fitness, *args, **kwargs):
    max_gen = kwargs['max_gen']
    return generation >= max_gen


def tournament_selection(population, fitness, number_parents, *args, **kwargs):
    t = kwargs['tournament_size']
    selected = []

    for _ in range(number_parents):
        # Seleccionar aleatoriamente t individuos y evaluar su fitness
        participants = random.sample(list(zip(population, fitness)), t)
        # Seleccionar el mejor participante
        winner = max(participants, key=lambda x: x[1])
        selected.append(winner[0])  # Añadir el mejor individuo al resultado

    return selected


def one_point_crossover(parent1, parent2, p_cross, *args, **kwargs):
    if random.random() < p_cross:
        point = random.randint(1, len(parent1) - 1)  # Punto de cruce
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    else:
        # Si no ocurre cruce, los hijos son copias de los padres
        child1, child2 = parent1[:], parent2[:]
    return child1, child2


def uniform_mutation(chromosome, p_mut, *args, **kwargs):
    dataset = kwargs['dataset']
    n_days = dataset["n_days"]
    n_hours_day = dataset["n_hours_day"]
    alphabet = list(range(n_days * n_hours_day))  # Todas las franjas posibles

    mutated = chromosome[:]
    for i in range(len(mutated)):
        if random.random() < p_mut:
            mutated[i] = random.choice(alphabet)  # Cambia a una hora aleatoria

    return mutated


def generational_replacement(population, fitness, offspring, fitness_offspring, *args, **kwargs):
    # Combina población actual y descendientes
    combined_population = population + offspring
    combined_fitness = fitness + fitness_offspring

    # Ordenar por fitness (mayor es mejor)
    sorted_population = [x for _, x in sorted(zip(combined_fitness, combined_population), reverse=True)]
    sorted_fitness = sorted(combined_fitness, reverse=True)

    # Selecciona los mejores individuos hasta completar el tamaño original
    new_population = sorted_population[:len(population)]
    new_fitness = sorted_fitness[:len(population)]

    return new_population, new_fitness


#USO COMPLETO DEL ALGORITMO
# Configuración del dataset
dataset = {
    "n_courses": 3,
    "n_days": 3,
    "n_hours_day": 3,
    "courses": [("IA", 1), ("ALG", 2), ("BD", 3)]
}

# Parámetros
pop_size = 100
offspring_size = 50
p_cross = 0.8
p_mut = 0.1
max_gen = 200

# Llamada al algoritmo
population, fitness, generation, best_fitness, mean_fitness = genetic_algorithm(
    generate_initial_population_timetabling,
    pop_size,
    fitness_timetabling,
    generation_stop,
    offspring_size,
    tournament_selection,
    one_point_crossover,
    p_cross,
    uniform_mutation,
    p_mut,
    generational_replacement,
    dataset=dataset,
    tournament_size=5,
    max_gen=max_gen
)

print(f"Mejor fitness alcanzado: {max(best_fitness)}")
print(f"Generaciones ejecutadas: {generation}")
"""
