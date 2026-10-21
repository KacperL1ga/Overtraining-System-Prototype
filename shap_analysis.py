import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

polish_feature = {
    'hrv': 'HRV (ms)',
    'resting_heart_rate': 'Tętno spoczynkowe (bpm)',
    'hrv_baseline': 'Norma bazowa HRV',
    'rhr_baseline': 'Norma bazowa RHR',
    'hrv_deviation': 'Odchylenie HRV',
    'rhr_deviation': 'Odchylenie RHR',
    'hrv_7d_mean': 'Średnie HRV (7 dni) ',
    'rhr_7d_mean': 'Średnia RHR (7 dni)',
    'sleep_hours': 'Czas snu',
    'sleep_hours_7d_mean': 'Średnia czas snu (7 dni)',
    'sleep_efficiency': 'Efektywnosc snu',
    'deep_sleep_hours': 'Sen głęboki',
    'rem_sleep_hours': 'Sen REM',
    'day_strain': 'Obciążenie dnia',
    'activity_strain': 'Obciążenie treningowe',
    'calories_burned': 'Spalone kalorie',
    'skin_temp_deviation': 'Odchylenie temp. ciała',
    'workout_completed': 'Trening wykonany'
}

def shap_section(model, input_data: pd.DataFrame):
    st.subheader("Dlaczego taki wynik? Analiza SHAP")
    st.caption("Wyjaśnienie, które z wprowadzanych parametrów miały znaczący wpływ na Revovery Score")

    # Inicjalizacja TreeExplainera od SHAP i wyliczanie wartości SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)

    # Utworzenie tabeli łączącej nazwy column z wartościami od SHAP
    df_shap = pd.DataFrame({
        'features': input_data.columns,
        'shap_values': shap_values.values[0]
    }).sort_values(by = 'shap_values', ascending = False)   # Sortowanie parametrów od najistotniejszego

    # Podział wartości na dodatnie i ujemne
    df_plus = df_shap[df_shap['shap_values'] > 0].head(3)
    df_minus = df_shap[df_shap['shap_values'] < 0].tail(3).iloc[::-1]

    # Podział sekcji na kolumny z plusami i minusami
    col_plus, col_minus = st.columns(2)

    with col_plus:
        st.markdown("**📈 Top 3 parametry podnoszące wynik**")
        if df_plus.empty:
            st.info("Brak parametrów o wyraźnie dodatnim wpływie")
        else:
            for _, row in df_plus.iterrows():
                val = input_data[row['features']].values[0]
                # Podmiana nazwy kolumny na polski odpowiednik lub pozostawienie oryginału (klucz, wartosc)
                feature_name = polish_feature.get(row['features'], row['features'])
                st.metric(
                    label = feature_name,                   # Nazwa cechy
                    value = f"{val:.1f}",                   # Wartosć cechy z formularza
                    delta = f"+{row['shap_values']:.2f}"    # Wpływ SHAP w procentach
                )

    with col_minus:
        st.markdown("**📉 Top 3 parametry pogarszające wynik**")
        if df_minus.empty:
            st.info("Brak parametrów o wyraźnie negatywnym  wpływie")
        else:
            for _, row in df_minus.iterrows():
                val = input_data[row['features']].values[0]
                # Podmiana nazwy kolumny na polski odpowiednik lub pozostawienie oryginału (klucz, wartosc)
                feature_name = polish_feature.get(row['features'], row['features'])
                st.metric(
                    label = feature_name,                   # Nazwa cechy
                    value = f"{val:.1f}",                   # Wartosć cechy z formularza
                    delta = f"{row['shap_values']:.2f}"     # Wpływ SHAP w procentach
                )

    # Rozwijana sekcja z pełnym wykresem wodospadowym SHAP
    with st.expander("Szczegółowy wykres wodospadowy", expanded = False):

        st.markdown("""
                ### 📖 Jak czytać wykres wodospadowy?
                * **📊 Oś pozioma $f(x)$**: Wynik regeneracji wyrazony w punktach procentowych.
                * **🎯 Wartość bazowa $E[f(x)]$**: średni wynik regeneracji w bazie danych.
                * **🔴 Kolor czerwony**: Paski skierowane w prawo - cechy które **podniosły** wartość.
                * **🔵 Kolor niebieski**: Paski skierowane w lewo - cechy które **obniżyły** wartość.
                * **🏷️ Wartości przy słupkach ($+X.XX$ / $-X.XX$)**: Dokładna liczba punktów dodanych / odjętych przez dany parametr.
                """)

        # Tworzymy nowa kartę
        fig, ax = plt.subplots(figsize = (8, 5))

        # Rysujemy wykres wodospadowy bez wyświetlania w oknie
        shap.plots.waterfall(shap_values[0], show = False)

        # Przekazanie do Streamlit aktualnie rysowaną figurę i wyświetlamy
        st.pyplot(plt.gcf())

        # Czyszczenie pamięci podręcznej matplotlib
        plt.close()