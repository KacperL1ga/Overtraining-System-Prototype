import pandas as pd

from database import get_user_history

def calculate_baseline_deviation(user_id: str, curr_hrv: float, curr_rhr: float):
    # Pobieranie historii danego uzytkownika z bazy
    df_history = get_user_history(user_id)

    # Do wyliczenia potrzebne są co najmniej 3 dni w bazie danych
    if len(df_history) < 3:                                     # W przypadku zbyt małej ilości danych funkcja nic nie zwraca
        return None

    if len(df_history) < 30:                                    # Po 3 dniach obliczane są odchylenia od baseline z poprzednich dni
        hrv_baseline = df_history['hrv'].mean()
        rhr_baseline = df_history['resting_heart_rate'].mean()
    else:
        hrv_baseline = df_history['hrv'].tail(30).mean()        # Co miesiąc baseline jest aktualizowany jako średnia z 30 dni
        rhr_baseline = df_history['resting_heart_rate'].tail(30).mean()

    # Obliczanie odchyleń od baseline'u
    hrv_deviation = curr_hrv - hrv_baseline
    rhr_deviation = curr_rhr - rhr_baseline

    # Zwracamy odchylenia do dalszych obliczeń oraz baseline do zapisu w bazie
    return hrv_deviation, rhr_deviation, hrv_baseline, rhr_baseline

def input_features_table(user_id: str, current_data: dict):
    # Pobieranie historii danego użytkownika z bazy danych

    df_history = get_user_history(user_id)
    df_current = pd.DataFrame([current_data])

    # Połącznie danych historycznych z dzisiejszymi
    if not df_history.empty:
        df_combined = pd.concat([df_history, df_current], ignore_index=True)
    else:
        df_combined = df_current

    # Usuwanie duplikaty daty z zachowaniem najnowszego wpisu
    df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')

    # Sortowanie danych chronologiczne wedlug daty
    df_combined['date'] = pd.to_datetime(df_combined['date'])
    df_combined = df_combined.sort_values(by='date')

    # Obliczanie średnich z ostatniego tygodnia
    recent_7 = df_combined.tail(7)
    hrv_7d_mean = recent_7['hrv'].mean()
    rhr_7d_mean = recent_7['resting_heart_rate'].mean()
    sleep_hours_7d_mean = recent_7['sleep_hours'].mean()

    # Wektor 18 cech przeznaczony do trenowania w modelu
    feature_dict = {
        'hrv': current_data['hrv'],
        'hrv_7d_mean': hrv_7d_mean,
        'resting_heart_rate': current_data['resting_heart_rate'],
        'rhr_7d_mean': rhr_7d_mean,
        'hrv_baseline': current_data['hrv_baseline'],
        'rhr_baseline': current_data['rhr_baseline'],
        'skin_temp_deviation': current_data['skin_temp_deviation'],
        'hrv_deviation': current_data['hrv_deviation'],
        'rhr_deviation': current_data['rhr_deviation'],
        'sleep_hours': current_data['sleep_hours'],
        'sleep_hours_7d_mean': sleep_hours_7d_mean,
        'sleep_efficiency': current_data['sleep_efficiency'],
        'deep_sleep_hours': current_data['deep_sleep_hours'],
        'rem_sleep_hours': current_data['rem_sleep_hours'],
        'day_strain': current_data['day_strain'],
        'activity_strain': current_data['activity_strain'],
        'calories_burned': current_data['calories_burned'],
        'workout_completed': current_data['workout_completed'],
    }
    return pd.DataFrame([feature_dict])

# Funkcja służąca obliczaniu stosunku ACWR
def calculate_acwr(user_id: str, curr_day_strain: float) -> float:
    df_history = get_user_history(user_id)

    # Pobranie historii z walidacją brakujących wierszy
    if not df_history.empty and 'day_strain' in df_history.columns:
        past_strain = df_history['day_strain'].dropna().tolist()
    else:
        past_strain = []

    # Dołącznie aktualnego day_strain do zestawienia
    past_strain.append(curr_day_strain)
    all_strain = pd.Series(past_strain)

    acute_strain = all_strain.tail(7).mean()            # Obciążenie ostre - srednia z 7 dni
    chronic_strain = all_strain.tail(28).mean()         # Obciążenie przewlekłe - srednia z 28 dni

    acwr = acute_strain / chronic_strain if chronic_strain > 0 else 1.0
    return round(acwr, 2)       # Wynik zaokrąglamy do 2 miejsc po przecinku

def interpret_acwr(acwr_value: float):
    if acwr_value < 0.8:
        return "Zbyt mało treningów", "info", "🔵"
    elif acwr_value <= 1.3:
        return "Optymalne obciążenie", "success", "🟢"
    elif acwr_value <= 1.5:
        return "Ostrzeżenie (wysokie obciążenie)", "warning", "🟡"
    else:
        return "Wysokie ryzyko przetrenowania!", "error", "🔴"