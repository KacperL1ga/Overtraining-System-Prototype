import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Wczytanie danych
file_path = 'data/whoop_fitness_dataset_100k.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['user_id', 'date'])

# Inżynieria cech (Feature Engineering)
# Odchylenie HRV i RHR od osobistej normy użytkownika
df['hrv_deviation'] = df['hrv'] - df['hrv_baseline']
df['rhr_deviation'] = df['resting_heart_rate'] - df['rhr_baseline']

# Łączna długość kluczowych faz snu (Głęboki + REM)
df['restorative_sleep_hours'] = df['deep_sleep_hours'] + df['rem_sleep_hours']

# Cechy opóźnione (wysiłek i sen z poprzdniego dnia)
df['prev_day_strain'] = df.groupby('user_id')['day_strain'].shift(1)
df['prev_sleep_hours'] = df.groupby('user_id')['sleep_hours'].shift(1)

# Czyszczenie wierszy z brakujacymi danymi
# Czyszczenie danych z brakujących danych
df_cleaned = df.dropna().copy()
print(f"Liczba wierszy po czyszczeniem: {len(df_cleaned)}")

# Wybór istotnych kolumn
features_cols = [
    'hrv', 'resting_heart_rate', 'hrv_baseline', 'rhr_baseline',
    'respiratory_rate', 'skin_temp_deviation', 'hrv_deviation',
    'rhr_deviation', # Fizjologia i Kardiologia

    'sleep_hours', 'sleep_efficiency', 'deep_sleep_hours',
    'rem_sleep_hours', 'wake_ups', 'prev_sleep_hours',
    'restorative_sleep_hours',  # Sen

    'day_strain', 'activity_strain', 'activity_duration_min',
    'calories_burned', 'workout_completed', 'prev_day_strain'  # Obciążenia i aktywność
]

print("Przykładowe dane: \n", df_cleaned.head())
print("Przykładowe dane: \n", df_cleaned.tail())


# Przygotowanie danych do uczenia
features = df_cleaned[features_cols]
target = df_cleaned['recovery_score']

# Podział danych na testowe i treningowe
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Inicjalizacja algorytmu RandomForest
#randomF = RandomForestRegressor(n_estimators=100, random_state=42)
#randomF.fit(X_train, y_train)

# Inicjalizacja XGBoost
# xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# xgb.fit(X_train, y_train)

# Przygotowanie danych pod algorytm Sieci Neuronowych
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicjalizacja algorytmu Sieci Neuronowych
mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Predykcje na testowym zbiorze danych
prediction = mlp.predict(X_test_scaled)

# Ocena stworzonego modelu
r2 = r2_score(y_test, prediction)
mea = mean_absolute_error(y_test, prediction)
rmse = np.sqrt(mean_squared_error(y_test, prediction))
print(f"R2 Score: {r2}")
print(f"Mean absolute error: {mea}")
print(f"Mean squared error: {rmse}")