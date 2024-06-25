import pandas as pd
import numpy as np
import random
import math

# Функція для розрахунку відстані між двома координатами використовуючи формулу Гарвіна
def calculate_distance(coord1, coord2):
    R = 6371  # Радіус Землі в км
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

# Функція для розрахунку евклідової відстані між двома координатами
def calculate_distance_euclidean(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    distance = math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    return distance

# Функція для розрахунку загальної відстані маршруту
def total_route_distance(route, locations):
    total_distance = 0
    for i in range(len(route)):
        loc1 = route[i]
        loc2 = route[(i + 1) % len(route)]
        total_distance += calculate_distance(locations[loc1]['coords'], locations[loc2]['coords'])
    return total_distance

# Функція для розрахунку пристосованості маршруту (fitness function)
def fitness(route, locations):
    total_distance = total_route_distance(route, locations)
    total_stay_time = sum(locations[loc]['stay_duration'] for loc in route)
    return total_distance + total_stay_time

# Функція для створення початкової популяції
def create_initial_population(size, locations):
    population = []
    for _ in range(size):
        route = list(locations.keys())
        random.shuffle(route)
        population.append(route)
    return population

# Функція для кросоверу (ordered crossover) між двома батьківськими маршрутами
def ordered_crossover(parent1, parent2):
    child = [None] * len(parent1)
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child[start:end] = parent1[start:end]
    p2_index = 0
    for i in range(len(child)):
        if child[i] is None:
            while parent2[p2_index] in child:
                p2_index += 1
            child[i] = parent2[p2_index]
    return child

# Функція для мутації (swap mutation) маршруту
def swap_mutation(route):
    idx1, idx2 = random.sample(range(len(route)), 2)
    route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

# Функція генетичного алгоритму для оптимізації маршруту
def genetic_algorithm(locations, population_size=100, generations=500):
    population = create_initial_population(population_size, locations)
    for _ in range(generations):
        population = sorted(population, key=lambda route: fitness(route, locations))
        new_population = population[:2]  # Елітизм: збереження найкращих маршрутів
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:50], 2)  # Вибір батьків з топ-50%
            child = ordered_crossover(parent1, parent2)
            child = swap_mutation(child)
            new_population.append(child)
        population = new_population
    best_route = sorted(population, key=lambda route: fitness(route, locations))[0]
    return best_route
