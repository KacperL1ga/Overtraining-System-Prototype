import sys
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


def get_zone(score):
    """Funkcja pomocnicza konwertująca wynik numeryczny na strefę regeneracji Whoop."""
    if score < 34:
        return "Czerwona 🔴"
    elif score < 67:
        return "Żółta 🟡"
    else:
        return "Zielona 🟢"


# =========================================================================
# KROK 1: Wczytanie i przygotowanie danych oraz inżynieria cech
# =========================================================================
print("📥 Wczytywanie i przygotowywanie danych...")
file_path = 'data/whoop_fitness_dataset_100k.csv'
df = pd.read_csv(file_path)

# Konwersja na typ datetime i sortowanie chronologiczne dla użytkowników
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['user_id', 'date'])

# Inżynieria cech (Feature Engineering)
df['hrv_deviation'] = df['hrv'] - df['hrv_baseline']
df['rhr_deviation'] = df['resting_heart_rate'] - df['rhr_baseline']

df['hrv_7d_mean'] = df.groupby('user_id')['hrv'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['rhr_7d_mean'] = df.groupby('user_id')['resting_heart_rate'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['sleep_hours_7d_mean'] = df.groupby('user_id')['sleep_hours'].transform(lambda x: x.rolling(7, min_periods=1).mean())

df['strain_7d_mean'] = df.groupby('user_id')['day_strain'].transform(lambda x: x.rolling(7, min_periods=1).mean())
df['strain_28d_mean'] = df.groupby('user_id')['day_strain'].transform(lambda x: x.rolling(28, min_periods=1).mean())
df['acwr'] = np.where(df['strain_28d_mean'] > 0, df['strain_7d_mean'] / df['strain_28d_mean'], 1.0)

if 'workout_completed' not in df.columns:
    df['workout_completed'] = (df['activity_strain'] > 0).astype(int)

features_cols = [
    'hrv', 'hrv_7d_mean', 'resting_heart_rate', 'rhr_7d_mean', 'skin_temp_deviation',
    'hrv_deviation', 'rhr_deviation',  # Fizjologia
    'sleep_hours', 'sleep_hours_7d_mean', 'sleep_efficiency', 'deep_sleep_hours',
    'rem_sleep_hours',  # Sen
    'day_strain', 'activity_strain', 'calories_burned', 'workout_completed', 'acwr'  # Obciążenia i ACWR
]

df_cleaned = df.dropna(subset=features_cols + ['recovery_score']).copy()

# =========================================================================
# KROK 2: Dwuetapowy podział danych 80(80/20)/20 na poziomie użytkowników
# =========================================================================
unique_users = df_cleaned['user_id'].unique().tolist()

# 1. PIERWSZY PODZIAŁ: 80% Pula Deweloperska / 20% Walidacja Końcowa
dev_users, val_users = train_test_split(unique_users, test_size=0.2, random_state=42)

df_dev = df_cleaned[df_cleaned['user_id'].isin(dev_users)].copy()
df_val = df_cleaned[df_cleaned['user_id'].isin(val_users)].copy()

# 2. DRUGI PODZIAŁ: Pula dev (80%) dzielona na 80% Trening i 20% Test Wewnętrzny
train_users, test_users = train_test_split(dev_users, test_size=0.2, random_state=42)

df_train = df_dev[df_dev['user_id'].isin(train_users)].copy()
df_test_internal = df_dev[df_dev['user_id'].isin(test_users)].copy()

X_train = df_train[features_cols]
y_train = df_train['recovery_score']

X_test_internal = df_test_internal[features_cols]
y_test_internal = df_test_internal['recovery_score']

X_val = df_val[features_cols]
y_val = df_val['recovery_score']
y_val_zones = [get_zone(s) for s in y_val]

print(f"\nStruktura podziału użytkowników:")
print(f" -> Zbiór Treningowy (64% całości):       {len(train_users)} użytkowników ({len(df_train)} wierszy)")
print(f" -> Zbiór Testowy Wewnętrzny (16% dev):   {len(test_users)} użytkowników ({len(df_test_internal)} wierszy)")
print(f" -> Zbiór Walidacji Końcowej (20% val):   {len(val_users)} użytkowników ({len(df_val)} wierszy)")

# =========================================================================
# KROK 3: Przeszukiwanie siatki hiperparametrów dla Regresji Ridge
# =========================================================================
print("\n" + "=" * 80)
print("⚙️ PRZESZUKIWANIE HIPERPARAMETRÓW REGRESJI RIDGE")
print("=" * 80)

# Definicja testowanych wariantów parametru alpha oraz algorytmów rozwiązujących (solver)
alpha_candidates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
solver_candidates = ['auto', 'svd', 'cholesky', 'lsqr', 'sag']

results = []
trained_pipelines = {}

# Pętla testująca kombinacje
run_id = 1
for alpha in alpha_candidates:
    for solver in solver_candidates:
        # Tworzymy Pipeline zapobiegający przeciekowi danych (StandardScaler fitowany na X_train)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=alpha, solver=solver, random_state=42))
        ])

        start_time = time.time()

        # Trenujemy model na zbiorze 64% (X_train)
        pipeline.fit(X_train, y_train.values)
        elapsed = time.time() - start_time

        # Weryfikacja jakości na wewnętrznym zbiorze testowym dev (16%)
        preds_int = pipeline.predict(X_test_internal)
        r2_int = r2_score(y_test_internal, preds_int)
        mae_int = mean_absolute_error(y_test_internal, preds_int)
        rmse_int = np.sqrt(mean_squared_error(y_test_internal, preds_int))

        # Zapisujemy do wyników
        config_key = f"alpha={alpha}_solver={solver}"
        trained_pipelines[config_key] = pipeline

        results.append({
            "Run": run_id,
            "Alpha": alpha,
            "Solver": solver,
            "R2 (Test Dev)": round(r2_int, 6),
            "MAE (Test Dev)": round(mae_int, 3),
            "RMSE (Test Dev)": round(rmse_int, 3),
            "Czas (s)": round(elapsed, 3)
        })
        run_id += 1

# =========================================================================
# KROK 4: Zestawienie i wybór najlepszego zestawu hiperparametrów
# =========================================================================
results_df = pd.DataFrame(results).sort_values(by="R2 (Test Dev)", ascending=False).reset_index(drop=True)

print("\n📊 TOP 10 NAJLEPSZYCH KONFIGURACJI RIDGE (NA ZBIORZE TEST DEV):")
print(results_df.head(10).to_string(index=False))

# Wybór zwycięskiego modelu
best_row = results_df.iloc[0]
best_alpha = best_row['Alpha']
best_solver = best_row['Solver']
best_config_key = f"alpha={best_alpha}_solver={best_solver}"
best_model = trained_pipelines[best_config_key]

print("\n" + "=" * 80)
print(f"🏆 NAJLEPSZA KONFIGURACJA: Alpha = {best_alpha} | Solver = {best_solver}")
print(f"   Wygenerowane R2 na teście dev: {best_row['R2 (Test Dev)']}")
print("=" * 80)

# =========================================================================
# KROK 5: Ostateczny sprawdzian zwycięzcy na niewidzianej Walidacji Końcowej (20%)
# =========================================================================
print("\n🚀 OSTATECZNA EVALUACJA NA ZBIORZE WALIDACJI KOŃCOWEJ (20% NIEWIDZIANYCH LUDZI)...")

final_preds = best_model.predict(X_val)
r2_val = r2_score(y_val, final_preds)
mae_val = mean_absolute_error(y_val, final_preds)
rmse_val = np.sqrt(mean_squared_error(y_val, final_preds))

final_pred_zones = [get_zone(s) for s in final_preds]
acc_val = accuracy_score(y_val_zones, final_pred_zones)

print(f"\n📈 METRYKI NA ZBIORZE WALIDACYJNYM:")
print(f" -> R2 Score:                      {r2_val:.4f}")
print(f" -> Średni błąd bezwzględny (MAE): {mae_val:.2f} punktów")
print(f" -> Pierwiastek błędu średn. (RMSE): {rmse_val:.2f} punktów")
print(f" -> Dokładność trafiania w strefy: {acc_val * 100:.2f}%")

# Zapis ostatecznego, optymalnego modelu pod aplikację Streamlit
joblib.dump(best_model, 'recovery_model.pkl')
print("\n✅ Najlepsza zoptymalizowana wersja modelu Ridge została zapisana jako 'recovery_model.pkl'!")