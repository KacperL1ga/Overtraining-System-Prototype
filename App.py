import streamlit as st
import pandas as pd
import numpy as np
import joblib
from IPython.utils.coloransi import value

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
st.divider()

# Przygotowywanie formularzu wejściowego
col1, col2, col3 = st.columns(3)

# Lewa kolumna przechowująca wartości kardiologiczne i fizjologiczne
with col1:
    st.subheader("🫀 Układ krążenia i fizjologia")
    hrv = st.number_input("Aktualne HRV (ms)", min_value=10.0, max_value=250.0, value=75.0, step=1.0)
    hrv_baseline = st.number_input("HRV Bazowe / Norma (ms)", min_value=10.0, max_value=250.0, value=80.0, step=1.0)

# Środkowa kolumna przechowująca wartości snu
with col2:
    st.subheader("🛏️ Sen i wypoczynek")
    sleep_efficiency = st.slider("Efektywność snu (%)", min_value=50.0, max_value=100.0, value=88.0, step=1.0)

# Prawa kolumna przechowująca wartośći obciążenia i aktywności
with col3:
    st.subheader("💪 Obciążenia i Aktywność")
    workout_completed = st.selectbox("Czy wykonano trening?", [1,0], format_func=lambda x: 'Tak' if x == 1 else 'Nie')

# Automatyczna inżynieria cech
hrv_deviation = hrv - hrv_baseline


# Przygotowanie danych wejściowych
features_cols = [
    'hrv', 'resting_heart_rate', 'hrv_baseline', 'rhr_baseline',
    'respiratory_rate', 'skin_temp_deviation', 'hrv_deviation',
    'rhr_deviation', # Fizjologia i Kardiologia

    'sleep_hours', 'sleep_efficiency', 'deep_sleep_hours',
    'rem_sleep_hours', 'wake_ups', 'prev_sleep_hours',
    'restorative_sleep_hours',  # Sen

    'day_strain', 'activity_strain', 'activity_duration_min',
    'calories_burned', 'workout_completed', 'prev_day_strain'  # Obciążenia i aktywność
]

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