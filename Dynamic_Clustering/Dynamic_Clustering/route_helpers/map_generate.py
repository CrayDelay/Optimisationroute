import folium
import pandas as pd
import numpy as np


# Функція для отримання кольору на основі номера кластера
def get_color(cluster_num):
    colors = ['red', 'blue', 'black', 'green', 'purple', 'orange', 'darkred',
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
              'gray', 'lightgray']
    return colors[cluster_num % len(colors)]


# Функція для створення карти з кластерами та збереження її у html файлі
def create_map(places, col_name, map_name="cluster_map.html"):
    avg_lat = places['lat'].mean()  # Середнє значення широти для центрування карти
    avg_lon = places['long'].mean()  # Середнє значення довготи для центрування карти
    mymap = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)  # Створення об'єкта карти з початковим зумом 6
    for idx, row in places.iterrows():
        popup_text = f"{row['name']}"  # Текст, який буде показаний при натисканні на маркер
        folium.CircleMarker(location=[row['lat'], row['long']],
                            radius=5,  # Розмір кругового маркера
                            color=get_color(row[col_name]),  # Колір кругового маркера
                            fill=True,
                            fill_color=get_color(row[col_name]),  # Колір заповнення маркера
                            popup=popup_text).add_to(mymap)  # Додавання маркера до карти

    mymap.save(map_name)  # Збереження карти у html файлі
