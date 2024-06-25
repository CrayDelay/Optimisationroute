import math
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def calculate_distance(coord1, coord2):
    R = 6371  #  Радіус Землі в км
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance


def calculate_centroid(locations):
    latitudes, longitudes = zip(*[locations[loc]['coords'] for loc in locations])
    return (sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes))


# Функція для коригування кластерів з урахуванням обмеження на щоденну тривалість
def adjust_clusters(day_locations, locations, daily_limit_hours):
    # Обчислення центроїдів для кожного дня
    cluster_centroids = {day: calculate_centroid({loc: locations[loc] for loc in locs}) for day, locs in
                         day_locations.items()}
    adjusted_day_locations = day_locations.copy()

    # Проходження по кожному дню та його локаціям
    for day, locs in day_locations.items():
        total_time = sum(locations[loc]['stay_duration'] for loc in locs)

        # Поки загальний час перевищує щоденне обмеження
        while total_time > daily_limit_hours:
            loc_to_move = None
            closest_cluster_day = None
            min_distance = float('inf')

            # Пошук найкращої локації для переміщення, виходячи з близькості до інших кластерів
            for loc in locs:
                for other_day, centroid in cluster_centroids.items():
                    if other_day != day:
                        distance = calculate_distance(locations[loc]['coords'], centroid)
                        if distance < min_distance:
                            min_distance = distance
                            loc_to_move = loc
                            closest_cluster_day = other_day

            # Переміщення вибраної локації до найближчого кластера
            if loc_to_move:
                adjusted_day_locations[day].remove(loc_to_move)
                adjusted_day_locations[closest_cluster_day].append(loc_to_move)

            # Перерахунок загального часу для поточного дня
            total_time = sum(locations[loc]['stay_duration'] for loc in adjusted_day_locations[day])

    return adjusted_day_locations
