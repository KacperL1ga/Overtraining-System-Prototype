import streamlit as st
import pandas as pd
import numpy as np
import joblib

from analytics import calculate_baseline_deviation
from database import init_db, get_user_history, delete_daily_log

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

# Przygotowywanie formularzu wejściowego
col1, col2, col3 = st.columns(3)

# Lewa kolumna przechowująca wartości kardiologiczne i fizjologiczne
with col1:
    st.subheader("🫀 Układ krążenia i fizjologia")
    hrv = st.number_input("Aktualne HRV (ms)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
    rhr = st.number_input("Aktualne RHR (bpm)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
    respiratory_rate = st.number_input("Częstość oddechów (oddechy/min)", min_value=10, max_value=25, value=15, step=1.0)
    skin_temp_deviation = st.number_input("Odchylenie temp. ciała (°C)", min_value=-3.0, max_value=5.0, value=0.0, step=0.1)

# Środkowa kolumna przechowująca wartości snu
with col2:
    st.subheader("🛏️ Sen i wypoczynek")
    sleep_hours = st.number_input("Czas snu (h)", min_value=0.0, max_value=15.0, value=8.0, step=0.5)
    sleep_efficiency = st.slider("Efektywność snu (%)", min_value=50.0, max_value=100.0, value=88.0, step=1.0)
    deep_sleep_hours = st.number_input("Sen głęboki (h)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    rem_sleep_hours = st.number_input("Sen REM (h)", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    restorative_sleep_hours = st.number_input("Sen regeneracyjny (h)", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
    prev_sleep_hours = st.number_input("Sen poprzedniej nocy (h)", min_value=0.0, max_value=15.0, value=8.0, step=0.5)
    wake_ups = st.number_input("Liczba przebudzeń", min_value=0.0, max_value=15, value=2, step=1)

# Prawa kolumna przechowująca wartośći obciążenia i aktywności
with col3:
    st.subheader("💪 Obciążenia i Aktywność")
    day_strain = st.slider("Obciążenie dnia (Day Strains)", min_value=0.0, max_value=21.0, value=12.0, step=0.1)
    activity_strain = st.slider("Obciążenie treningowe", min_value=0.0, max_value=21.0, value=8.0, step=0.1)
    activity_duration_min = st.number_input("Czas aktywności (min)", min_value=0, max_value=300, value=45, step=5)
    calories_burned = st.number_input("Spalone kalorie (kcal)", min_value=0, max_value=5000, value=1000, step=50)
    workout_completed = st.selectbox("Czy wykonano trening?", [1,0], format_func=lambda x: 'Tak' if x == 1 else 'Nie')
    prev_day_strain = st.slider("Obciążenie dnia poprzedniego", min_value=0.0, max_value=21.0, value=10.0, step=0.1)

# Przygotowanie danych wejściowych
features_cols = [
    'hrv', 'resting_heart_rate', 'hrv_baseline', 'rhr_baseline',
    'respiratory_rate', 'skin_temp_deviation', 'hrv_deviation',
    'rhr_deviation',                                            # Fizjologia i Kardiologia

    'sleep_hours', 'sleep_efficiency', 'deep_sleep_hours',
    'rem_sleep_hours', 'wake_ups', 'prev_sleep_hours',
    'restorative_sleep_hours',                                  # Sen

    'day_strain', 'activity_strain', 'activity_duration_min',
    'calories_burned', 'workout_completed', 'prev_day_strain'  # Obciążenia i aktywność
]

result = calculate_baseline_deviation("test_user_1", hrv, rhr)

if result is None:
    st.info("Wskaźnik regeneracji wymaga co najmniej 3 dni pomiarów.")
else:
    hrv_deviation, rhr_deviation, hrv_baseline, rhr_baseline = result

    input_data = pd.DataFrame([{
        'hrv': hrv,
        'resting_heart_rate': rhr,
        'hrv_baseline': hrv_baseline,
        'rhr_baseline': rhr_baseline,
        'respiratory_rate': respiratory_rate,
        'skin_temp_deviation': skin_temp_deviation,
        'hrv_deviation': hrv_deviation,
        'rhr_deviation': rhr_deviation,
        'sleep_hours': sleep_hours,
        'sleep_efficiency': sleep_efficiency,
        'deep_sleep_hours': deep_sleep_hours,
        'rem_sleep_hours': rem_sleep_hours,
        'wake_ups': wake_ups,
        'prev_sleep_hours': prev_sleep_hours,
        'restorative_sleep_hours': restorative_sleep_hours,
        'day_strain': day_strain,
        'activity_strain': activity_strain,
        'activity_duration_min': activity_duration_min,
        'calories_burned': calories_burned,
        'workout_completed': workout_completed,
        'prev_day_strain': prev_day_strain
    }])[features_cols]

    # Przycisk akcji i wyświetlanie wyniku
    if st.button("Wyznacz wskaźnik regeneracji", type="primary", use_container_width=True):
        prediction = model.predict(input_data)[0]
        score = float(np.clip(prediction, 0.0, 100.0))

        col_res1, col_res2 = st.columns([1,2])

        with col_res1:
            st.metric(label="Przewidywany Recovery Score", value=f"{score:.1f}%")

        with col_res2:
            if score >= 67:
                st.success("🟢 **Strefa Zielona (67-100%)**: Wysoka gotowość organizmu.")
            elif score >= 34:
                st.warning("🟡 **Strefa Żółta (34-66%)**: Średnia regeneracja.")
            else:
                st.error("🔴 **Strefa Czerwona (0-33%)**: Niska regeneracja.")

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
        st.dataframe(df_filtred, use_container_width=True)

        # Sekcja usuwania danych
        st.markdown("🗑️ Usuwanie wybranych wpisów")
        del_date = st.date_input("Wybierz datę do usunięcia wpisu", key="del_date")
        if st.button("Usuń wpis", type="secondary"):
            delete_daily_log(user_id, str(del_date))
            st.success(f"Usunięto wpis z dnia {del_date}. Odśwież stronę, aby zobaczyć zmiany.")
