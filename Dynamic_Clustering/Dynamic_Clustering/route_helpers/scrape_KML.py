import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

def get_places_table(file_name):
    kml_filename = file_name  # Ім'я KML файлу
    locations = []  # Порожній список для збереження місць
    with open(kml_filename, "r") as file:
        kml_input = file.readlines()  # Зчитування всіх рядків з файлу
        kml_input = "".join(kml_input)  # Об'єднання рядків в один рядок
        bs_kml_input = BeautifulSoup(kml_input, "xml")  # Парсинг XML за допомогою BeautifulSoup
        placemarks = bs_kml_input.findAll('Placemark')  # Знаходження всіх елементів 'Placemark'
        for placemark in placemarks:
            coords = placemark.find('coordinates').text.strip()  # Отримання координат
            long = coords.split(',')[0]  # Виділення довготи
            lat = coords.split(',')[1]  # Виділення широти
            locations.append({
                'name': placemark.find('name').text.strip(),  # Отримання назви місця
                'lat': lat,  # Збереження широти
                'long': long  # Збереження довготи
            })
    locations = pd.DataFrame(locations)  # Створення DataFrame з локацій
    locations['lat'] = locations['lat'].astype(float)  # Конвертація широти в тип float
    locations['long'] = locations['long'].astype(float)  # Конвертація довготи в тип float
    locations["coords"] = locations.apply(lambda row: (row["lat"], row["long"]), axis=1)  # Створення нового стовпця з координатами у вигляді кортежів

    return locations  # Повернення DataFrame з місцями
