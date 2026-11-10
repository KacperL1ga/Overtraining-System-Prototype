import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def get_zone(score):
    """Funkcja konwertująca wynik numeryczny na strefę regeneracji Whoop."""
    if score < 34:
        return "Czerwona 🔴"
    elif score < 67:
        return "Żółta 🟡"
    else:
        return "Zielona 🟢"


# Wczytanie i przygotowanie danych
# =================================
print("📥 Wczytywanie i przygotowywanie danych...")
file_path = 'data/whoop_fitness_dataset_100k.csv'
df = pd.read_csv(file_path)

# Konwersja daty i chronologiczne sortowanie dla użytkowników
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by = ['user_id', 'date'])

# Inżynieria cech (Feature Engineering)
# Odchylenie parametrów od osobistej normy bazowej
df['hrv_deviation'] = df['hrv'] - df['hrv_baseline']
df['rhr_deviation'] = df['resting_heart_rate'] - df['rhr_baseline']

# Wyliczenie średnich kroczących (7-dniowych) dla każdego użytkownika
df['hrv_7d_mean'] = df.groupby('user_id')['hrv'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['rhr_7d_mean'] = df.groupby('user_id')['resting_heart_rate'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['sleep_hours_7d_mean'] = df.groupby('user_id')['sleep_hours'].transform(lambda x: x.rolling(7, min_periods=1).mean())

# Obliczanie średnich obciążeń do ACWR (Acute:Chronic Workload Ratio)
df['strain_7d_mean'] = df.groupby('user_id')['day_strain'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['strain_28d_mean'] = df.groupby('user_id')['day_strain'].transform(lambda x: x.rolling(28, min_periods=1).mean())
df['acwr'] = np.where(df['strain_28d_mean'] > 0, df['strain_7d_mean'] / df['strain_28d_mean'], 1.0)

# Uzupełnienie flagi wykonania treningu, jeśli nie istnieje
if 'workout_completed' not in df.columns:
    df['workout_completed'] = (df['activity_strain'] > 0).astype(int)

# Spójny wektor 17 cech używany w całym prototypie (App.py, analytics.py, testy)
features_cols = [
    'hrv', 'hrv_7d_mean', 'resting_heart_rate', 'rhr_7d_mean', 'skin_temp_deviation',
    'hrv_deviation', 'rhr_deviation',                 # Fizjologia i Kardiologia

    'sleep_hours', 'sleep_hours_7d_mean', 'sleep_efficiency', 'deep_sleep_hours',
    'rem_sleep_hours',                                                       # Sen

    'day_strain', 'activity_strain', 'calories_burned', 'workout_completed', 'acwr'  # Obciążenia, aktywność i ACWR
]

# Czyszczenie wierszy z brakującymi danymi
df_cleaned = df.dropna(subset=features_cols + ['recovery_score']).copy()

## Dwuetapowy podział danych (80/20 po użytkownikach)
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


# Skalowanie danych (wymagane dla Ridge, MLP, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_internal_scaled = scaler.transform(X_test_internal)
X_val_scaled = scaler.transform(X_val)

# Definicja słownika z modelami do przetestowania
# ================================================
models = {
    "Ridge (Liniowy)": (Ridge(alpha=0.01), True),
    "MLP (Sieć Neuronowa)": (MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=42), True),
    "XGBoost": (XGBRegressor(max_depth=4, n_estimators=400, learning_rate=0.03, random_state=42, n_jobs=-1), False),
    "Random Forest": (RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1), False),
    "Extra Trees": (ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1), False),
    "K-Nearest Neighbors": (KNeighborsRegressor(n_neighbors=15, n_jobs=-1), True)
}

# Pętla testująca
# ===============
results = []
print(f"\n🚀 Rozpoczynam testowanie {len(models)} modeli na zbiorze {len(X_train)} wierszy treningowych...\n")

for name, (model, requires_scale) in models.items():
    start_time = time.time()

    # Trenujemy na danych przeskalowanych lub surowych
    if requires_scale:
        model.fit(X_train_scaled, y_train.values)
        preds_val = model.predict(X_val_scaled)
        preds_int = model.predict(X_test_internal_scaled)
    else:
        model.fit(X_train, y_train.values)
        preds_val = model.predict(X_val)
        preds_int = model.predict(X_test_internal)

    # Czas wykonania
    elapsed = time.time() - start_time

    # Metryki numeryczne na zbiorze walidacyjnym (Walidacja Końcowa)
    r2_val = r2_score(y_val, preds_val)
    r2_int = r2_score(y_test_internal, preds_int)
    mae_val = mean_absolute_error(y_val, preds_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))

    # Metryki kategoryczne dla właściwego zbioru (y_val vs preds_val)
    val_preds_zones = [get_zone(s) for s in preds_val]
    acc_val = accuracy_score(y_val_zones, val_preds_zones)

    results.append({
        "Model": name,
        "R2 Score": round(r2_val, 4),
        "Dokładność Stref (%)": round(acc_val * 100, 2),
        "MAE": round(mae_val, 2),
        "RMSE": round(rmse_val, 2),
        "Czas (s)": round(elapsed, 1)
    })
    print(f" -> Zakończono: {name:28s} w {round(elapsed, 1):4.1f}s | R2: {r2_val:.4f}")

# Prezentacja wyników
# ====================
comparison_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print("\n" + "=" * 80)
print("🏆 WYNIKI BENCHMARKU MODELI (17 CECH) 🏆")
print("=" * 80)
print(comparison_df.to_string(index=False))
print("=" * 80)
