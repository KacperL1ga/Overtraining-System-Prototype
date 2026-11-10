import pytest
import pandas as pd
import numpy as np
import joblib

from analytics import interpret_acwr, calculate_acwr, input_features_table
from database import save_daily_logs, get_user_history, delete_user_profile, rename_user_profile, get_all_users

### Testy jednostkowe - Analiza ACWR ###

def test_interpret_acwr_zones():
    # Sprawdzamy poprawność kategoryzacji wskaźnika ACWR
    # Strefa 1: Niska aktywność (< 0,8)
    text, status, emoji = interpret_acwr(0.65)
    assert status == "info"
    assert emoji == "🔵"
    assert "Zbyt mało" in text

    # Strefa 2: Sweet Spot (0.8 - 1.3)
    text, status, emoji = interpret_acwr(1.15)
    assert status == "success"
    assert emoji == "🟢"
    assert "Optymalne" in text

    # Strefa 3: Ostrzeżenie (1.31 - 1.5)
    text, status, emoji = interpret_acwr(1.42)
    assert status == "warning"
    assert emoji == "🟡"
    assert "Ostrzeżenie" in text

    # Strefa 4: Zagrożenie kontuzją (> 1.5)
    text, status, emoji = interpret_acwr(1.85)
    assert status == "error"
    assert emoji == "🔴"
    assert "Wysokie ryzyko" in text

def test_interpret_acwr_boundary_values():
    # Sprawdza zachowanie na dokładnych granicach przedziałów.
    assert interpret_acwr(0.8)[1] == "success"   # 0.8 to już Sweet Spot
    assert interpret_acwr(1.3)[1] == "success"   # 1.3 to jeszcze Sweet Spot
    assert interpret_acwr(1.5)[1] == "warning"   # 1.5 to granica ostrzeżenia
    assert interpret_acwr(1.51)[1] == "error"    # powyżej 1.5 to strefa czerwona


### Testy jednostkowe — Wektor cech ###

def test_input_features_table_structure():
    # Upewnia się, że funkcja generuje dokładnie 17 cech wymaganych przez model.
    mock_data = {
        'date': '2026-11-01',
        'hrv': 65.0,
        'resting_heart_rate': 55.0,
        'hrv_baseline': 60.0,
        'rhr_baseline': 54.0,
        'skin_temp_deviation': 0.1,
        'hrv_deviation': 5.0,
        'rhr_deviation': 1.0,
        'sleep_hours': 7.5,
        'sleep_efficiency': 88.0,
        'deep_sleep_hours': 1.5,
        'rem_sleep_hours': 1.8,
        'day_strain': 12.0,
        'activity_strain': 8.5,
        'calories_burned': 2100,
        'workout_completed': 1
    }
    # Dodaje do wektora 7-dniowe średnie i acwr
    df_out = input_features_table("test_user_dummy", mock_data)

    # Model oczekuje dokładnie 17 kolumn
    assert df_out.shape[1] == 17
    assert 'hrv_7d_mean' in df_out.columns
    assert 'rhr_7d_mean' in df_out.columns
    assert 'sleep_hours_7d_mean' in df_out.columns
    assert 'acwr' in df_out.columns
    assert not df_out.isna().any().any(), "Wektor cech nie może zawierać wartości NaN!"


### Testy integracyjne - Baza danych ###
@pytest.fixture
def temp_user():
    # Fixture tworzący unikalnego użytkownika do testów i sprzątający po teście.
    user_name = "pytest_athlete_999"
    yield user_name
    # Sprzątanie po wykonaniu testu
    delete_user_profile(user_name)
    delete_user_profile(user_name + "_renamed")

def test_database_crud_lifecycle(temp_user):
    # Testuje pełen cykl życia w bazie: zapis -> odczyt -> zmiana nazwy -> usunięcie.
    # 1. Zapis pomiaru
    log_entry = {
        "user_id": temp_user,
        "date": "2026-11-01",
        "hrv": 80.0,
        "resting_heart_rate": 50.0,
        "skin_temp_deviation": 0.0,
        "hrv_baseline": 75.0,
        "rhr_baseline": 52.0,
        "hrv_deviation": 5.0,
        "rhr_deviation": -2.0,
        "sleep_hours": 8.0,
        "sleep_efficiency": 90.0,
        "deep_sleep_hours": 2.0,
        "rem_sleep_hours": 2.0,
        "day_strain": 10.0,
        "activity_strain": 6.0,
        "calories_burned": 2000,
        "workout_completed": 1,
        "recovery_score": 85.0
    }
    save_daily_logs(log_entry)

    # 2. Odczyt i weryfikacja
    history = get_user_history(temp_user)
    assert len(history) == 1
    assert history.iloc[0]['hrv'] == 80.0
    assert history.iloc[0]['recovery_score'] == 85.0

    # 3. Zmiana nazwy profilu
    new_name = temp_user + "_renamed"
    rename_user_profile(temp_user, new_name)
    assert len(get_user_history(temp_user)) == 0
    assert len(get_user_history(new_name)) == 1

    # 4. Usunięcie profilu
    delete_user_profile(new_name)
    assert len(get_user_history(new_name)) == 0

def test_unique_constraint_overwrites_same_date(temp_user):
    # Sprawdzamy, czy wpisanie pomiaru z tą samą datą nadpisuje rekord, zamiast tworzyć duplikat.
    base_entry = {
        "user_id": temp_user,
        "date": "2026-11-01",
        "hrv": 70.0, "resting_heart_rate": 55.0, "skin_temp_deviation": 0.0,
        "hrv_baseline": 70.0, "rhr_baseline": 55.0, "hrv_deviation": 0.0, "rhr_deviation": 0.0,
        "sleep_hours": 7.0, "sleep_efficiency": 80.0, "deep_sleep_hours": 1.0, "rem_sleep_hours": 1.0,
        "day_strain": 10.0, "activity_strain": 5.0, "calories_burned": 1800, "workout_completed": 1,
        "recovery_score": 60.0
    }
    save_daily_logs(base_entry)

    # Nadpisujemy ten sam dzień innym recovery_score
    updated_entry = base_entry.copy()
    updated_entry["recovery_score"] = 92.5
    save_daily_logs(updated_entry)
    history = get_user_history(temp_user)
    assert len(history) == 1, "Powinien istnieć dokładnie 1 rekord dla danego dnia!"
    assert history.iloc[0]['recovery_score'] == 92.5


### Testy integracyjne - model regeneracji ###
def test_model_prediction_range_and_stability():
    # Sprawdzamy, czy zapisany model poprawnie przyjmuje dane z 17 cechami i generuje wynik w granicach 0-100%.
    model = joblib.load('recovery_model.pkl')
    mock_input = pd.DataFrame([{
        'hrv': 75.0, 'hrv_7d_mean': 72.0,
        'resting_heart_rate': 58.0, 'rhr_7d_mean': 59.0,
        'skin_temp_deviation': 0.0, 'hrv_deviation': 1.0, 'rhr_deviation': 0.0,
        'sleep_hours': 8.0, 'sleep_hours_7d_mean': 7.8, 'sleep_efficiency': 85.0,
        'deep_sleep_hours': 1.8, 'rem_sleep_hours': 1.7,
        'day_strain': 11.5, 'activity_strain': 7.0, 'calories_burned': 2200,
        'workout_completed': 1,
        'acwr': 1.05
    }])
    pred = model.predict(mock_input)[0]
    score = float(np.clip(pred, 0.0, 100.0))
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0, f"Recovery score poza zakresem: {score}"
