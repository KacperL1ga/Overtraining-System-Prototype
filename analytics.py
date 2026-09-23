from database import get_user_history

def calculate_baseline_deviation(user_id: str, curr_hrv: float, curr_rhr: float):
    # Pobieranie historii danego uzytkownika z bazy
    df_history = get_user_history(user_id)

    # Do wyliczenia potrzebne są co najmniej 3 dni w bazie danych
    if len(df_history) < 3:                                     # W przypadku zbyt małej ilości danych funkcja nic nie zwraca
        return None

    if len(df_history) < 30:                                    # Po 3 dnia obliczane ją odchylenia od baseline z poprzednich dni
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