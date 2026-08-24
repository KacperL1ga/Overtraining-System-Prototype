import pandas as pd
import numpy as np

# Wczytanie danych
file_path = 'data/whoop_fitness_dataset_100k.csv'
df = pd.read_csv(file_path)

# Wybór istotnych kolumn
good_columns = [
    'recovery_score',   # cel

    'hrv', 'resting_heart_rate', 'hrv_baseline', 'rhr_baseline',
    'respiratory_rate', 'skin_temp_deviation',  # Fizjologia i Kardiologia

    'sleep_hours', 'sleep_efficiency', 'deep_sleep_hours',
    'rem_sleep_hours', 'wake_ups',  # Sen

    'day_strain', 'activity_strain', 'activity_duration_min',
    'calories_burned', 'workout_completed'
]

# Przycięcie ramki danych do istotnych kolumn
df_model = df[good_columns].copy()
