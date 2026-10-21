import streamlit as st
import pandas as pd
import numpy as np
import time

from database import get_user_history, delete_daily_log, save_daily_logs, delete_users_profile, rename_user_profile
from analytics import calculate_baseline_deviation, input_features_table

# Funckja konwertująca wartości na float
def floatt(val, default=0.0):
    try:
        return float(val) if pd.notnull(val) else default
    except (ValueError, TypeError):
        return default

# Funkcja przedstawiająca formularz do edycji wpisów w bazie danych
@st.dialog("✏️ Edycja pomiarów", width="large")
def show_edit_dialog(user_id: str, action_date, model):
    date_str = str(action_date)
    df_history = get_user_history(user_id)

    # Wczytanie wiersza dla wyznaczonej daty
    df_entry = df_history[df_history['date'].astype(str) == date_str]
    if df_entry.empty:
        st.warning(f"Brak wpisów w bazie danych dla daty {date_str}.")
        return

    # W przypadku braku danych uzupełniamy je wartością 0.0
    row = df_entry.fillna(0.0).iloc[0]
    old_score = floatt(row.get('recovery_score', 0.0))      # Pobranie starego recovery score do późniejszego porównania

    st.write(f"Modyfikuj wpis z dnia: **{date_str}**")

    with st.form("edit_dialog_form"):           # Formularz z wartościami bazowymi pobranymi z bazy danych
        col1, col2,col3 = st.columns(3)
        with col1:
            st.subheader("🫀 Układ krążenia i fizjologia")
            hrv = st.number_input("HRV (ms)", min_value=10.0, max_value=250.0, value=floatt(row['hrv'], 75.0), step=1.0)
            rhr = st.number_input("RHR (bpm)", min_value=10.0, max_value=250.0, value=floatt(row['resting_heart_rate'], 60.0), step=1.0)
            skin_temp_deviation = st.number_input("Odchylenie temp. ciała (°C)", min_value=-3.0, max_value=5.0,value=floatt(row['skin_temp_deviation'] ,0.0), step=0.1)

        # Środkowa kolumna przechowująca wartości snu
        with col2:
            st.subheader("🛏️ Sen i wypoczynek")
            sleep_hours = st.number_input("Czas snu (h)", min_value=0.0, max_value=15.0, value=floatt(row['sleep_hours'], 8.0), step=0.1)
            sleep_efficiency = st.slider("Efektywność snu (%)", min_value=50.0, max_value=100.0, value=floatt(row['sleep_efficiency'], 88.0), step=1.0)
            deep_sleep_hours = st.number_input("Sen głęboki (h)", min_value=0.0, max_value=5.0, value=floatt(row['deep_sleep_hours'], 1.5), step=0.1)
            rem_sleep_hours = st.number_input("Sen REM (h)", min_value=0.0, max_value=5.0, value=floatt(row['rem_sleep_hours'], 1.5), step=0.1)

        # Prawa kolumna przechowująca wartośći obciążenia i aktywności
        with col3:
            st.subheader("💪 Obciążenia i Aktywność")
            day_strain = st.slider("Obciążenie dnia (Day Strain)", min_value=0.0, max_value=21.0, value=floatt(row['day_strain'], 12.0), step=0.1)
            activity_strain = st.slider("Obciążenie treningowe (Activity Strain)", min_value=0.0, max_value=21.0,value=floatt(row['activity_strain'], 8.0), step=0.1)
            calories_burned = st.number_input("Spalone kalorie (kcal)", min_value=0, max_value=6000, value=int(row['calories_burned']), step=50)

        submitted = st.form_submit_button("Zapisz zmiany i przelicz ponownie", type="primary")

    if submitted:       # Po zatwierdzeniu zmiany przestępujemy do ponownego wylicznenia recovery score
        workout_completed = 1 if activity_strain > 0.0 else 0
        res = calculate_baseline_deviation(user_id, hrv, rhr)       # Ponowne obliczanie odchyleń i baseline'ów
        hrv_deviation, rhr_deviation, hrv_baseline, rhr_baseline = res if res else (0.0, 0.0, hrv, rhr)

        current_data = {
            'hrv': hrv, 'resting_heart_rate': rhr, 'hrv_baseline': hrv_baseline, 'rhr_baseline': rhr_baseline,
            'skin_temp_deviation': skin_temp_deviation, 'hrv_deviation': hrv_deviation, 'rhr_deviation': rhr_deviation,
            'sleep_hours': sleep_hours, 'sleep_efficiency': sleep_efficiency, 'deep_sleep_hours': deep_sleep_hours,
            'rem_sleep_hours': rem_sleep_hours, 'day_strain': day_strain, 'activity_strain': activity_strain,
            'calories_burned': calories_burned, 'workout_completed': workout_completed
        }

        input_data  = input_features_table(user_id, current_data)       # Tworzenie DataFrame do predykcji i zapisu w bazie danych
        new_score = floatt(np.clip(model.predict(input_data)[0], 0.0, 100.0))   # Uzycie modelu to predykcji nowego recovery_score

        log_data = current_data.copy()
        log_data.update({"user_id": user_id, "date": date_str, "recovery_score": new_score})    # Uzupełniene DataFrame o brakujące dane

        save_daily_logs(log_data)           # Zapis pomiarów do bazy danych

        # Przy pomyślnym zapisie dostaniemy komunikat z informacją o różnicy w recovery_score
        st.success("Pomyślnie zaktualizowano dane!")
        st.metric(
            label = "Zaktualizowany Recovery Score",
            value = f"{new_score:.2f}%",
            delta = f"{new_score - old_score:+.2f}% względem poprzedniego"
        )

        time.sleep(2) # Po 3 sekundach następuje przeładowanie strony
        st.rerun()

@st.dialog("⚠️ Potwierdzenie usunięcia")
def show_delete_dialog(user_id: str, action_date):
    st.warning(f"Czy na pewno chcesz bezpowrotnie usunąć pomiary z dnia **{action_date}**?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Tak, usuń", type="primary", use_container_width=True):
            delete_daily_log(user_id, str(action_date))
            st.success("Wpis został usunięty!")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("Anuluj", use_container_width=True):
            st.rerun()

@st.dialog("⚠️ Usunięcie profilu")
def show_delete_profile_dialog(user_id: str):
    st.error(f"Czy na pewno chcesz bezpowrotnie usunąć profil **{user_id}** wraz ze wszystkimi jego pomiarami?")
    st.caption("Tej operacji nie można cofnąć!")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Tak, usuń wszystko", type = "primary", width = "stretch"):
            delete_users_profile(user_id)
            st.success("Profil został usunięty!")
            time.sleep(1)
            st.rerun()

    with col2:
        if st.button("Anuluj", width = "stretch"):
            st.rerun()

@st.dialog("🔄 Zmiana nazwy profilu")
def show_rename_profile_dialog(old_user_id: str):
    new_name = st.text_input("Nowa nazwa profilu:", value = old_user_id).strip()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Zapisz", type = "primary", width = "stretch"):
            if not new_name:
                st.error("Nazwa nie może być pusta!")
            elif new_name == old_user_id:
                st.warning("Nowa nazwa jest identyczna jak obecna.")
            else:
                rename_user_profile(old_user_id, new_name)
                st.success("Nazwa została zmieniona!")
                time.sleep(1)
                st.rerun()

    with col2:
        if st.button("Anuluj", width = "stretch"):
            st.rerun()