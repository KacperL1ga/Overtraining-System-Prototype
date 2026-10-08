import streamlit as st
import pandas as pd
import numpy as np
import joblib

from analytics import calculate_baseline_deviation, input_features_table, calculate_acwr, interpret_acwr
from database import init_db, get_user_history, save_daily_logs
from dialogs import show_delete_dialog, show_edit_dialog
from charts import render_charts_section
from charts2 import chart_physiology

# Inicjalizacja bazy danych
init_db()

# Konfiguracja układu strony
st.set_page_config(
    page_title="Predykcja przetrenowania i regeneracji organizmu",
    page_icon="📊📉",
    layout="wide"
)

# Wczytanie zapisanego modelu z pamięci podręcznej
@st.cache_resource
def load_model():
    return joblib.load('xgb_model.pkl')

model = load_model()

# Tytuł strony i struktura kolumnowa
st.title("Kalkulator Regeneracji Organizmu i Niwelowania Przetrenowania")
st.write("Wprowadź dane biometryczne oraz parametry snu i treningu aby wyznaczyć prognozowany wskaźnik regeneracji")

user_id = st.text_input("Nazwa użytkownika", value="test_user_1")
st.write(f"Witaj, **{user_id}**! 👋")
st.divider()

# Pobranie histori z bazy danych dla aktualnego użytkownika
df_history = get_user_history(user_id)

# Sprawdzenie warunku czy użytkownik ma w bazie przynajmniej 3 wpisy
if len(df_history) < 3:
    st.info("Zanim zaczniesz korzystać z kalkulatora, musisz ustalić swoje wstępne Baseline. Wprowadź parametry z ostatnich 3 dni.")

    # Tworzenie interaktywnego formularza
    with st.form("onboarding_form"):
        st.subheader("Inicjalzacja danych z 3 dni")

        # Podział formularza na części dla każdego dnia
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            st.markdown("Dzień 1")
            date1 = st.date_input("Data (Dzień 1)")
            hrv1 = st.number_input("HRV (Dzień 1)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
            rhr1 = st.number_input("RHR (Dzień 1)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
        with col_d2:
            st.markdown("Dzień 2")
            date2 = st.date_input("Data (Dzień 2)")
            hrv2 = st.number_input("HRV (Dzień 2)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
            rhr2 = st.number_input("RHR (Dzień 2)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
        with col_d3:
            st.markdown("Dzień 3")
            date3 = st.date_input("Data (Dzień 3)")
            hrv3 = st.number_input("HRV (Dzień 3)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
            rhr3 = st.number_input("RHR (Dzień 3)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)

        # Przycisk do zatwierdzenia formularza
        submitted = st.form_submit_button("Aktualiacja bazy danych", type="primary")

        if submitted:
            # Walidacja danych, sprawdzanie czy daty idą rosnąco i chronologicznie
            if not (date1 < date2 < date3):
                st.error("Błąd: Daty muszą iść w porządku chronologicznym i bez powtarzania!")
            # Walidacja danych, sprawdzanie czy odstęp pomiędzy Dzień1 i Dzień3 nie są zbyt odległe
            elif (date3 - date1).days > 7:
                st.error("Błąd: Wpisane dane nie mogą być na przestrzeni większej niż 7 dni!")
            # Przy bez błędnej walidacji inicjalizujemy zapis do bazy danych
            else:
                base_data_template = {
                    "user_id": user_id,
                    "skin_temp_deviation": None, "hrv_baseline": None, "rhr_baseline": None,
                    "hrv_deviation": None, "rhr_deviation": None, "sleep_hours": None,
                    "sleep_efficiency": None, "deep_sleep_hours": None,
                    "rem_sleep_hours": None, "day_strain": None, "activity_strain": None, "calories_burned": None,
                    "workout_completed": None, "recovery_score": None
                }

                # Zapisujemy dane do bazy danych
                data1 = base_data_template.copy()
                data1.update({"date": date1, "hrv": hrv1, "resting_heart_rate": rhr1})
                save_daily_logs(data1)

                data2 = base_data_template.copy()
                data2.update({"date": date2, "hrv": hrv2, "resting_heart_rate": rhr2})
                save_daily_logs(data2)

                data3 = base_data_template.copy()
                data3.update({"date": date3, "hrv": hrv3, "resting_heart_rate": rhr3})
                save_daily_logs(data3)

                st.success("Gotowe! Można teraz odświeżyć stronę aby używać pełnego kalkulatora")

else:
    st.subheader("🗓️ Wybierz dzień pomiaru")
    curr_date = st.date_input("Data pomiaru")

    # Przygotowywanie formularzu wejściowego
    col1, col2, col3 = st.columns(3)

    # Lewa kolumna przechowująca wartości kardiologiczne i fizjologiczne
    with col1:
        st.subheader("🫀 Układ krążenia i fizjologia")
        hrv = st.number_input("Aktualne HRV (ms)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
        rhr = st.number_input("Aktualne RHR (bpm)", min_value=10.0, max_value=250.0, value=60.0, step=1.0)
        skin_temp_deviation = st.number_input("Odchylenie temp. ciała (°C)", min_value=-3.0, max_value=5.0, value=0.0, step=0.1)

    # Środkowa kolumna przechowująca wartości snu
    with col2:
        st.subheader("🛏️ Sen i wypoczynek")
        sleep_hours = st.number_input("Czas snu (h)", min_value=0.0, max_value=15.0, value=8.0, step=0.1)
        sleep_efficiency = st.slider("Efektywność snu (%)", min_value=50.0, max_value=100.0, value=88.0, step=1.0)
        deep_sleep_hours = st.number_input("Sen głęboki (h)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
        rem_sleep_hours = st.number_input("Sen REM (h)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)

    # Prawa kolumna przechowująca wartośći obciążenia i aktywności
    with col3:
        st.subheader("💪 Obciążenia i Aktywność")
        day_strain = st.slider("Obciążenie dnia (Day Strain)", min_value=0.0, max_value=21.0, value=12.0, step=0.1)
        activity_strain = st.slider("Obciążenie treningowe (Activity Strain)", min_value=0.0, max_value=21.0, value=8.0, step=0.1)
        calories_burned = st.number_input("Spalone kalorie (kcal)", min_value=0, max_value=6000, value=1000, step=50)

    # Automatyczne wyznaczanie odbytego treningu na podstawie obciążenia treningowego
    workout_completed = 1 if activity_strain > 0 else 0

    result = calculate_baseline_deviation(user_id, hrv, rhr)

    if result is None:
        st.info("Wskaźnik regeneracji wymaga co najmniej 3 dni pomiarów.")
    else:
        hrv_deviation, rhr_deviation, hrv_baseline, rhr_baseline = result

        current_data = {
            'hrv': hrv,
            'resting_heart_rate': rhr,
            'hrv_baseline': hrv_baseline,
            'rhr_baseline': rhr_baseline,
            'skin_temp_deviation': skin_temp_deviation,
            'hrv_deviation': hrv_deviation,
            'rhr_deviation': rhr_deviation,
            'sleep_hours': sleep_hours,
            'sleep_efficiency': sleep_efficiency,
            'deep_sleep_hours': deep_sleep_hours,
            'rem_sleep_hours': rem_sleep_hours,
            'day_strain': day_strain,
            'activity_strain': activity_strain,
            'calories_burned': calories_burned,
            'workout_completed': workout_completed
        }

        # Checkbox z zgodą na zapis parametrów do bazy
        save_do_db = st.checkbox("Chcę zapisać powyższe parametry do historii pomiarów", value=True)

        # Funkcja dodająca 3 parametry średnich i przygotowująca Tabele do precykcji w modelu
        input_data = input_features_table(user_id, current_data)

        # Przycisk akcji i wyświetlanie wyniku
        if st.button("Wyznacz wskaźnik regeneracji", type="primary", width="stretch"):

            # Walidacja wprowadzanych danych aby uniknąć wpisów z tą sama data
            df_history = get_user_history(user_id)
            date_exist = False
            if not df_history.empty and 'date' in df_history.columns:
                date_exist = str(curr_date) in df_history['date'].astype(str).values

            if date_exist and save_do_db:
                st.warning(f"Wpis dla daty **{curr_date}** już istnieje w bazie danych! Aby go zmodyfikować skorzystaj z modyfikacji w sekcji historii pomiarów")
            else:
                prediction = model.predict(input_data)[0]
                score = float(np.clip(prediction, 0.0, 100.0))

                if save_do_db:
                    log_data = {
                        "user_id": user_id,
                        "date": str(curr_date),
                        "hrv": hrv,
                        "resting_heart_rate": rhr,
                        "skin_temp_deviation": skin_temp_deviation,
                        "hrv_baseline": hrv_baseline,
                        "rhr_baseline": rhr_baseline,
                        "hrv_deviation": hrv_deviation,
                        "rhr_deviation": rhr_deviation,
                        "sleep_hours": sleep_hours,
                        "sleep_efficiency": sleep_efficiency,
                        "deep_sleep_hours": deep_sleep_hours,
                        "rem_sleep_hours": rem_sleep_hours,
                        "day_strain": day_strain,
                        "activity_strain": activity_strain,
                        "calories_burned": calories_burned,
                        "workout_completed": workout_completed,
                        "recovery_score": score
                    }
                    save_daily_logs(log_data)
                    st.info(f"Zapisano pomiary z dnia {curr_date} w bazie danych.")

                acwr_value = calculate_acwr(user_id, day_strain)
                status_text, status_type, status_emoji = interpret_acwr(acwr_value)

                col_res1, col_res2, col_res3 = st.columns([1,1,1.2])

                with col_res1:
                    st.metric(label="Przewidywany Recovery Score", value=f"{score:.2f}%")
                    if score >= 67:
                        st.success("🟢 **Strefa Zielona (67-100%)**")
                    elif score >= 34:
                        st.warning("🟡 **Strefa Żółta (34-66%)**")
                    else:
                        st.error("🔴 **Strefa Czerwona (0-33%)**")

                with col_res2:
                    st.metric(label="Wskaźnik ACWR (7d / 28d)", value=f"{acwr_value}")
                    st.caption("Stosunek obciążenia ostrego do przewlekłego")

                with col_res3:
                    st.markdown("**Ocena ryzyka przetrenowania**")
                    if status_type == "info":
                        st.info(f"{status_emoji} {status_text}")
                    elif status_type == "success":
                        st.success(f"{status_emoji} {status_text}")
                    elif status_type == "warning":
                        st.warning(f"{status_emoji} {status_text}")
                    else:
                        st.error(f"{status_emoji} {status_text}")

    st.divider()
    with st.expander("📊 Twoja historia pomiarów", expanded=False):
        df_history = get_user_history(user_id)

        if df_history.empty:
            st.info("Brak wpisów w bazie dla tego użytkownika.")
        else:
            # Konwersja w przypadku braku dat
            df_history['date'] = pd.to_datetime(df_history['date'])

            # Filtry dat
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                start_date = st.date_input("Data od", value=df_history['date'].min())
            with col_f2:
                end_date = st.date_input("Data do", value=df_history['date'].max())

            # Filtrowanie
            mask = (df_history['date'].dt.date >= start_date) & (df_history['date'].dt.date <= end_date)
            df_filtred = df_history.loc[mask].copy()

            # Formatowanie daty do wyświetlenia
            df_filtred['date'] = df_filtred['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(df_filtred, width="stretch")

            # Sekcja modtfikacji danych
            st.markdown("**🛠️ Modyfikacja wabranego wpisu**")
            action_date = st.date_input("Wybierz datę do edycji lub usunięcia", key="action_date")

            col_del, col_edit = st.columns(2)

            with col_del:
                if st.button("🗑️ Usuń wpis", width="stretch"):
                    show_delete_dialog(user_id, action_date)

            with col_edit:
                if st.button("✏️️ Edytuj wpis", type="primary", width="stretch"):
                    show_edit_dialog(user_id, action_date, model)
    st.divider()
    # render_charts_section(user_id, get_user_history(user_id))
    chart_physiology(df_history)