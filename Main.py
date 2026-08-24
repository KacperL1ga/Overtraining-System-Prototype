import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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
    'calories_burned', 'workout_completed'  # Obciążenia i aktywność
]

# Przycięcie ramki danych do istotnych kolumn
df_model = df[good_columns].copy()

# Inżynieria cech (Feature Engineering
# Odchylenie HRV i RHR od osobistej normy użytkownika
df_model['hrv_deviation'] = df_model['hrv'] - df_model['hrv_baseline']
df_model['rhr_deviation'] = df_model['resting_heart_rate'] - df_model['rhr_baseline']

# Łączna długość kluczowych faz snu (Głęboki + REM)
df_model['restorative_sleep_hours'] = df_model['deep_sleep_hours'] + df_model['rem_sleep_hours']

# Preprocessing danych
# Czyszczenie danych z brakujących danych
print(f"Liczba wierszy przed czyszczeniem: {len(df_model)}")
print(f"Liczba brakujących danych: {df_model.isna().sum()}")
missing_values = df_model.isna().sum()
df_cleaned = df_model.dropna()
print(f"Liczba wierszy po czyszczeniem: {len(df_cleaned)}")
print("Przykładowe dane: \n", df_cleaned.head())
print("Przykładowe dane: \n", df_cleaned.tail())

# Mapa Korelacji kolumn numerycznych
numeric_df = df_cleaned.select_dtypes(include=[np.number])
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(12,10))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Heatmapa korelacji danych numerycznych')
    plt.show()

corr_matrix = numeric_df.corr()
recovery_corr = corr_matrix['recovery_score'].sort_values(ascending=False)
print("Zestawienie korelacji z 'Recovery Score':\n", recovery_corr)