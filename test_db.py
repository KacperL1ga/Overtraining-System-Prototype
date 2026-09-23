from database import init_db, save_daily_logs, get_user_history

# Inicjalizacja bazy danych
init_db()

# Przygotowanie przykladowego rekordu
sample_data = {
    "user_id": "test_user_1",
    "date": "2026-09-07",
    "hrv": 65.5,
    "resting_heart_rate": 54.0,
    "skin_temp_deviation": 0.1,
    "hrv_baseline": 68.0,
    "rhr_baseline": 53.0,
    "hrv_deviation": -2.5,
    "rhr_deviation": 1.0,
    "sleep_hours": 7.5,
    "sleep_efficiency": 88.0,
    "deep_sleep_hours": 1.5,
    "rem_sleep_hours": 2.0,
    "day_strain": 12.4,
    "activity_strain": 15.0,
    "calories_burned": 2400.0,
    "workout_completed": 1,
    "recovery_score": 78.5
}

# Zapis danych do bazy
save_daily_logs(sample_data)

# Odczyt danych testowych
df = get_user_history("test_user_1")

# Wyswietlenie podsumowania testu
print(f"Liczba pobranych wierszy: {len(df)}")
print(f"Liczba kolumn w bazie: {len(df.columns)}")
print("\nFragment pobranych danych:")
print(df[["user_id", "date", "hrv", "resting_heart_rate", "recovery_score"]])