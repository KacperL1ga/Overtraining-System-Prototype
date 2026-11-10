import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error, mean_squared_error

def get_zone(score):
    # Funkcja pomocnicza konwertująca wynik numeryczny na strefę regeneracji Whoop.
    if score < 34:
        return "Czerwona 🔴"
    elif score < 67:
        return "Żółta 🟡"
    else:
        return "Zielona 🟢"


# Wczytanie i przygotowanie danych
# ==================================
print("Wczytywanie danych z pliku csv...")
file_path = 'data/whoop_fitness_dataset_100k.csv'
df = pd.read_csv(file_path)

# Konwersja na typ datetime i sortowanie chronologiczne dla użytkowników
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['user_id', 'date'])

# Inżynieria cech (Feature Engineering)
# Odchylenie HRV i RHR od osobistej normy użytkownika
df['hrv_deviation'] = df['hrv'] - df['hrv_baseline']
df['rhr_deviation'] = df['resting_heart_rate'] - df['rhr_baseline']

# Wyliczenie średnich 7-dniowych dla każdego użytkownika
df['hrv_7d_mean'] = df.groupby('user_id')['hrv'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['rhr_7d_mean'] = df.groupby('user_id')['resting_heart_rate'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['sleep_hours_7d_mean'] = df.groupby('user_id')['sleep_hours'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# Obliczanie średnich obciążeń do ACWR (Acute:Chronic Workload Ratio)
df['strain_7d_mean'] = df.groupby('user_id')['day_strain'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['strain_28d_mean'] = df.groupby('user_id')['day_strain'].transform(lambda x: x.rolling(28, min_periods=1).mean())
df['acwr'] = np.where(df['strain_28d_mean'] > 0, df['strain_7d_mean'] / df['strain_28d_mean'], 1.0)

# Uzupełnienie flagi wykonania treningu na podstawie obciążenia
if 'workout_completed' not in df.columns:
    df['workout_completed'] = (df['activity_strain'] > 0).astype(int)

# Wybór 17 kluczowych cech
features_cols = [
    'hrv', 'hrv_7d_mean', 'resting_heart_rate', 'rhr_7d_mean','skin_temp_deviation',
    'hrv_deviation', 'rhr_deviation',                 # Fizjologia i Kardiologia

    'sleep_hours', 'sleep_hours_7d_mean', 'sleep_efficiency', 'deep_sleep_hours',
    'rem_sleep_hours',                                                       # Sen

    'day_strain', 'activity_strain', 'calories_burned', 'workout_completed', 'acwr'  # Obciążenia, aktywność i ACWR
]

# Czyszczenie wierszy z brakującymi danymi
print(f"Liczba wierszy przed czyszczeniem: {len(df)}")
df_cleaned = df.dropna(subset=features_cols + ['recovery_score']).copy()
print(f"Liczba wierszy po czyszczeniu: {len(df_cleaned)}")

# Dwuetapowy podział danych (80/20 po użytkownikach)
# ====================================================
unique_users = df_cleaned['user_id'].unique().tolist()

# Pierwszy podział: 80% Deweloperski / 20% Walidacja ostateczna (nowi użytkownicy)
dev_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

df_dev = df_cleaned[df_cleaned['user_id'].isin(dev_users)].copy()
df_val = df_cleaned[df_cleaned['user_id'].isin(val_users)].copy()

# Drugi podział: Pula deweloperska (80%) dzielona wewnętrznie na 80% Trening i 20% Test Wewnętrzny
train_users, test_users = train_test_split(dev_users, test_size=0.2, random_state=42)

df_train = df_dev[df_dev['user_id'].isin(train_users)].copy()
df_test_internal = df_dev[df_dev['user_id'].isin(test_users)].copy()

# Przygotowanie danych wejściowych i etykiety dla zbioru treningowego
X_train = df_train[features_cols]
y_train = df_train['recovery_score']

# Przygotowanie danych wejściowych i etykiety dla zbioru testowego
X_test_internal = df_test_internal[features_cols]
y_test_internal = df_test_internal['recovery_score']

# Przygotowanie danych wejściowych i etykiety dla zbioru walidacyjnego (nowi użytkownicy)
X_val = df_val[features_cols]
y_val = df_val['recovery_score']
y_val_zones = [get_zone(s) for s in y_val]

print(f"\nStruktura podziału użytkowników:")
print(f" Całkowita liczba użytkowników:        {len(unique_users)}")
print(f" Zbiór Treningowy (64% całości):       {len(train_users)} użytkowników ({len(df_train)} wierszy)")
print(f" Zbiór Testowy Wewnętrzny (16% dev):   {len(test_users)} użytkowników ({len(df_test_internal)} wierszy)")
print(f" Zbiór Walidacji Końcowej (20% val):   {len(val_users)} użytkowników ({len(df_val)} wierszy)")

# Trenowanie modelu Ridge Pipeline i weryfikacja
# ================================================
print("Trenowanie modelu (17 CECH)")
print("=" * 50)

# Aby uniknąć wycieku informacji stosujemy Pipeline łącząc scalar z modelem
model_ridge_pipeline = Pipeline([
    ('scaler', StandardScaler()),                       # Standaryzacja cechy
    ('ridge', Ridge(alpha = 0.01, random_state = 42))   # Regresja liniowa z regularyzacją L2
])
# Dzięki temu do App.py wystarczy przekazać surowe DataFrame i wywołać predict()

start_time = time.time()
# Uczenie odbywa się na 80% z puli treningowej (X_train)
model_ridge_pipeline.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Model pomyślnie wytrenowany w czasie: {train_time:.3f} s")

# Wyniki na wewnętrznym zbiorze testowym (z puli treningowej)
preds_internal = model_ridge_pipeline.predict(X_test_internal)
r2_internal = r2_score(y_test_internal, preds_internal)
print(f"R2 Score na testowym wewnętrznym (pula dev): {r2_internal:.4f}")

# Ostateczny test na zbiorze walidacyjnym (nowi użytkownicy z 20% walidacji)
predictions = model_ridge_pipeline.predict(X_val)

r2 = r2_score(y_val, predictions)
mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

# Konwersja przewidywanych wyników na strefy
y_pred_zones = [get_zone(s) for s in predictions]
accuracy = accuracy_score(y_val_zones, y_pred_zones)

print(f"\nOstateczne wyniki modelu na zbiorze walidacyjnym (20% niewidzianych pomiarów ludzi):")
print(f" R2 Score:                      {r2:.4f}")
print(f" Średni błąd bezwzględny (MAE): {mae:.2f} punktów")
print(f" Pierwiastek błędu średn. (RMSE): {rmse:.2f} punktów")
print(f" Dokładność trafiania w strefy: {accuracy * 100:.2f}%\n")

print("Szczegółowy raport klasyfikacji stref (Zbiór Walidacyjny):")
print(classification_report(y_val_zones, y_pred_zones))

# Zapis wyuczonego modelu do pliku .pkl pod Streamlit
# =====================================================
print("Zapisanie najlepszego modelu do pliku .pkl")
print("=" * 50)

model_path = 'recovery_model.pkl'
joblib.dump(model_ridge_pipeline, model_path)
print(f"Model Ridge Pipeline zapisany jako '{model_path}'\n")

# Wykresy SHAP na próbce ze zbioru walidacyjnego
# ===============================================

print("Generowanie wykresów wyjaśnialności SHAP")
print("=" * 50)

# Do analizy bierzemy reprezentatywną próbkę zbioru testowego (maks 1000 obserwacji)
sample_size = min(1000, len(X_val))
X_val_sample = X_val.sample(n = sample_size, random_state = 42)

# Wyciągnięcie kroków z Pipeline'u
scaler_step = model_ridge_pipeline.named_steps['scaler']    # obliczanie sredniej i odchylen dla cech
ridge_step = model_ridge_pipeline.named_steps['ridge']      # wyciągniecie wyuczonych wag i wartości bazowych

# Dokładne wyliczenie wartości SHAP dla modelu liniowego ze standaryzacją
scaled_diff = (X_val_sample.values - scaler_step.mean_) / scaler_step.scale_    # macierz o wymiarach [liczba_probek x licza_cech]
shap_matrix = scaled_diff * ridge_step.coef_        # macierz reprezentujaca punktowy wpływ 17 cech na osteteczna predykcje

# Tworzenie obiektu wyjaśniającego SHAP do wykresów
shap_values = shap.Explanation(
    values = shap_matrix,                   # macierz wpływów cech
    base_values = np.full(sample_size, ridge_step.intercept_),  # startowa średnia predykcja punktowa
    data = X_val_sample.values,             # surowe i nieprzeskalowane wartosci parametrów biometrycznych
    feature_names = features_cols           # nazwy 17 kolumn wejściowych
)

# Wykres 1: Słupkowy wykres ważności cech
plt.figure(figsize = (10, 6))
shap.plots.bar(shap_values, max_display = 15, show = False)
plt.title("Ranking najważniejszych cech (Model: Ridge Pipeline)", fontsize = 12)
plt.tight_layout()
plt.savefig("shap_bar_winner.png", dpi = 150)
print("💾 Zapisano wykres: shap_bar_winner.png")

# Wykres 2: Wykres zbiorczy wpływu i kierunku cech na wynik regeneracji
plt.figure(figsize = (10, 6))
shap.summary_plot(shap_values, X_val_sample, feature_names = features_cols, max_display = 15, show = False)
plt.title("Wpływ poszczególnych parametrów na wskaźnik regeneracji", fontsize = 12)
plt.tight_layout()
plt.savefig("shap_summary_winner.png", dpi = 150)
print("💾 Zapisano wykres: shap_summary_winner.png")

# Wyświetlenie okien wykresów, jeśli uruchamiane w środowisku interaktywnym
if sys.stdout.isatty():
    plt.show()
else:
    plt.close('all')