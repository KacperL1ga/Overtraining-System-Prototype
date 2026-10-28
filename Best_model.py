import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

# Tworzymy katalog na zapisywane modele
os.makedirs("models_experiment", exist_ok=True)

# Wczytanie danych z pliku csv
file_path = 'data/whoop_fitness_dataset_100k.csv'
df = pd.read_csv(file_path)

# Konwersja na typ datetime i sortowanie chronologiczne dla uzytkowników
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['user_id', 'date'])

# Inżynieria cech (Feature Engineering)
# Odchylenie HRV i RHR od osobistej normy użytkownika
df['hrv_deviation'] = df['hrv'] - df['hrv_baseline']
df['rhr_deviation'] = df['resting_heart_rate'] - df['rhr_baseline']

# Wyliczenie średnich 7-dniowych dla każdego użytkownika
df['hrv_7d_mean'] = df.groupby('user_id')['hrv'].transform(lambda x: x.rolling(7, min_periods = 1).mean())
df['rhr_7d_mean'] = df.groupby('user_id')['resting_heart_rate'].transform(lambda x: x.rolling(7, min_periods = 1).mean())
df['sleep_hours_7d_mean'] = df.groupby('user_id')['sleep_hours'].transform(lambda x: x.rolling(7, min_periods = 1).mean())

if 'workout_completed' not in df.columns:
    df['workout_completed'] = (df['activity_strain'] > 0).astype(int)

# Wybór istotnych kolumn
features_cols = [
    'hrv', 'hrv_7d_mean', 'resting_heart_rate', 'rhr_7d_mean', 'hrv_baseline', 'rhr_baseline',
    'skin_temp_deviation', 'hrv_deviation', 'rhr_deviation',                 # Fizjologia i Kardiologia

    'sleep_hours', 'sleep_hours_7d_mean', 'sleep_efficiency', 'deep_sleep_hours',
    'rem_sleep_hours',                                                       # Sen

    'day_strain', 'activity_strain', 'calories_burned', 'workout_completed'  # Obciążenia i aktywność
]

# Czyszczenie wierszy z brakujacymi danymi
print(f"Liczba wierszy przed czyszczeniem: {len(df)}")
df_cleaned = df.dropna(subset = features_cols + ['recovery_score']).copy()
print(f"Liczba wierszy po czyszczeniu: {len(df_cleaned)}")

# -------------------------------------------------------------------------
# KROK 1: Podział na poziomie UŻYTKOWNIKÓW (80% Trening / 20% Walidacja)
# -------------------------------------------------------------------------
unique_users = df_cleaned['user_id'].unique().tolist()
print(f"Łączna liczba unikalnych użytkowników: {len(unique_users)}")

dev_users, val_users = train_test_split(unique_users, test_size = 0.2, random_state = 42)

df_dev = df_cleaned[df_cleaned['user_id'].isin(dev_users)].copy()
df_val = df_cleaned[df_cleaned['user_id'].isin(val_users)].copy()

print(f"Użytkownicy w zbiorze deweloperskim: {len(dev_users)} ({len(df_dev)} wierszy)")
print(f"Użytkownicy w zbiorze walidacyjnym: {len(val_users)} ({len(df_val)} wierszy)")

# Przygotowanie ostatecznego zbioru walidacyjnego (X_val, y_val)
X_val = df_val[features_cols]
y_val = df_val['recovery_score']

# Przygotowanie puli deweloperskiej
X_dev = df_dev[features_cols]
y_dev = df_dev['recovery_score']

# Funkcja pomocnicza do stref
def get_zone(score):
  if score < 34:
    return "Czerwona 🔴"
  elif score < 67:
    return "Żółta 🟡"
  else:
    return "Zielona 🟢"

y_val_zones = [get_zone(s) for s in y_val]

# -------------------------------------------------------------------------
# KROK 2: Pętla 10 iteracji – Uczenie 10 różnych modeli
# -------------------------------------------------------------------------
experiments_results = []
trained_models = {}

print("\n Rozpoczynam serię 10 experymentów...")

for i in range(1,11):
    seed = i * 100
    X_train, X_test, y_train, y_test = train_test_split(
        X_dev, y_dev, test_size = 0.2, random_state = seed
    )

    # Inicjalizacja i uczenie modelu XGBoost
    model = XGBRegressor(
        max_depth=4,  # Płytsze drzewa = lepsze uogólnianie na NOWYCH ludzi
        n_estimators=400,
        learning_rate=0.03,  # Wolniejsze, stabilniejsze uczenie
        subsample=0.8,  # Każde drzewo widzi tylko 80% wierszy (zapobiega overfittingowi)
        colsample_bytree=0.8,  # Każde drzewo losuje 80% cech
        reg_alpha=0.1,  # Regularyzacja L1
        reg_lambda=1.0,  # Regularyzacja L2
        random_state=seed,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 1. Wyniki na zbiorze testowym z danego losowania
    test_preds = model.predict(X_test)
    r2_test = r2_score(y_test, test_preds)

    # 2. Najważniejsze, wyniki na ściśle tajnym zbiorze walidacyjnym
    val_preds = model.predict(X_val)
    r2_val = r2_score(y_val, val_preds)
    mea_val = mean_absolute_error(y_val, val_preds)
    rmse_val = np.sqrt(mean_squared_error(y_val, val_preds))

    # Dokładność stref na zbiorze walidaycjnym
    val_pred_zones = [get_zone(s) for s in val_preds]
    acc_val = accuracy_score(y_val_zones, val_pred_zones) * 100

    # Zapisujemy model do pliku
    model_filename = f"models_experiment/xgb_model_run_{i}.pkl"
    joblib.dump(model, model_filename)
    trained_models[i] = model

    # Gromadzimy metryki do tabeli
    experiments_results.append({
        "Model": f"Iteracja_{i}",
        "Seed": seed,
        "R2 (Test 80/20)": round(r2_test, 4),
        "R2 (Walidacja 20%)": round(r2_val, 4),
        "MAE (Walidacja)": round(mea_val, 2),
        "RMSE (Walidacja)": round(rmse_val, 2),
        "Strefy Acc (%)": round(acc_val, 2),
        "Plik": model_filename
    })

    print(f" -> Iteracja {i:2d}/10 ukończona | R2 na walidacyjnym {r2_val:.4f} | Strefy: {acc_val:.1f}%")
# -------------------------------------------------------------------------
# KROK 3: Zestawienie wyników i wybór najlepszego modelu
# -------------------------------------------------------------------------

results_df = pd.DataFrame(experiments_results)

# Sortowanie według najlepszego wyniku R2 na zbiorze walidacyjnym
results_df = results_df.sort_values(by = "R2 (Walidacja 20%)", ascending = False).reset_index(drop = True)

print("\n" + "=" * 80)
print("📊 PODSUMOWANIE I RANKING 10 MODELI (ZBIÓR WALIDACYJNY):")
print("=" * 80)
print(results_df.to_string(index=False))

# Wybieramy zwycięzcę rankingu (wiersz 0 po posortowaniu)
best_run = results_df.iloc[0]
best_model_name = best_run['Model']
best_model_num = int(best_model_name.split("_")[1])
best_model = trained_models[best_model_num]
print("\n" + "=" * 80)
print(f"🏆 NAJLEPSZY MODEL: {best_model_name}")
print(f"   R2 na nowych użytkownikach: {best_run['R2 (Walidacja 20%)']}")
print(f"   Średni błąd MAE: {best_run['MAE (Walidacja)']} punktów")
print(f"   Trafność w strefy: {best_run['Strefy Acc (%)']}%")
print("=" * 80)

# Nadpisujemy główny plik 'xgb_model.pkl' najlepszym modelem dla aplikacji Streamlit
joblib.dump(best_model, 'xgb_model.pkl')
print("\n✅ Najlepszy model został zapisany jako 'xgb_model.pkl' i jest gotowy dla Streamlita!")