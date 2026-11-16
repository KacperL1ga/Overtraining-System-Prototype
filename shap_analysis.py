import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

polish_feature = {
    'hrv': 'HRV (ms)',
    'resting_heart_rate': 'Tętno spoczynkowe (bpm)',
    'hrv_deviation': 'Odchylenie HRV',
    'rhr_deviation': 'Odchylenie RHR',
    'hrv_7d_mean': 'Średnie HRV (7 dni)',
    'rhr_7d_mean': 'Średnie RHR (7 dni)',
    'sleep_hours': 'Czas snu',
    'sleep_hours_7d_mean': 'Średni czas snu (7 dni)',
    'sleep_efficiency': 'Efektywność snu',
    'deep_sleep_hours': 'Sen głęboki',
    'rem_sleep_hours': 'Sen REM',
    'day_strain': 'Obciążenie dnia',
    'activity_strain': 'Obciążenie treningowe',
    'calories_burned': 'Spalone kalorie',
    'skin_temp_deviation': 'Odchylenie temp. ciała',
    'workout_completed': 'Trening wykonany',
    'acwr': 'Wskaźnik obciążenia (ACWR)'
}

def shap_section(model, input_data: pd.DataFrame):
    st.subheader("Dlaczego taki wynik? Analiza SHAP")
    st.caption("Wyjaśnienie, które z wprowadzanych parametrów miały znaczący wpływ na Recovery Score")

    # Wyliczanie wartości SHAP dla modelu Ridge w Pipeline
    # Pipeline składa się z dwóch kroków: 'scaler' (StandardScaler) i 'ridge' (Ridge)
    scaler = model.named_steps['scaler']        # Wyciągamy scaler żeby odczytać mean_ i scale_ z treningu
    ridge = model.named_steps['ridge']          # Wyciągamy Ridge żeby odczytać coef_ (współczynniki) i intercept_ (wyraz wolny)

    x_vals = input_data.values                  # Surowe wartości wejściowe jako tablica numpy (1 wiersz x 19 cech)

    # Obliczamy o ile każda cecha odchyla się od średniej treningowej
    # Jest to dokładnie to, co robi StandardScaler przy predict
    scaled_diff = (x_vals - scaler.mean_) / scaler.scale_

    # Wartość SHAP każdej cechy = jej skalowane odchylenie * współczynnik Ridge
    # Dla modelu liniowego to wzór: phi_i = (x_i - mean_i) / scale_i * coef_i
    phi = scaled_diff * ridge.coef_

    # Przygotowanie danych do zsumowania
    feature_names = list(input_data.columns)
    phi_values = list(phi[0])
    raw_values = list(x_vals[0])

    feat_idx = {name: idx for idx, name in enumerate(feature_names)}

    # Definiowanie par do zsumowania: (cecha bazowa, odchylenie, nowa nazwa wyświetlana na ekranie)
    pairs_to_combine = [
        ('hrv', 'hrv_deviation', 'Zmiennosc rytmu serca'),
        ('resting_heart_rate', 'rhr_deviation', 'Tętno spoczynkowe'),
        ('day_strain', 'activity_strain', 'Obciązenie')
    ]

    for base_feat, dev_feat, new_name in pairs_to_combine:
        if base_feat in feat_idx and dev_feat in feat_idx:
            idx_base = feat_idx[base_feat]
            idx_dev = feat_idx[dev_feat]

            # Sumowanie wartości SHAP
            phi_values[idx_base] = phi_values[idx_base] + phi_values[idx_dev]

            # Zmiana nazwy na nową
            feature_names[idx_base] = new_name

            # Odznaczanie odcgyleń do usuniecia, aby nie dublowaly sie na wykresie SHAP
            phi_values[idx_dev] = None
            feature_names[idx_dev] = None
            raw_values[idx_dev] = None

    # Filtrowanie usuniętych cech, oznaczonych jako None
    final_phi = [p for p in phi_values if p is not None]
    final_names = [n for n in feature_names if n is not None]
    final_raw = [r for r in raw_values if r is not None]

    # Tłumaczenie wszystkich pozostałych nazw na polskie odpowiedniki
    final_names = [polish_feature.get(n, n) for n in final_names]

    # Konwertujemy na numpy array, co jest przyjazne dla SHAP
    new_phi = np.array([final_phi])
    new_x_vals = np.array([final_raw])

    # Budujemy obiekt Explanation przyjmowany przez wykresy SHAP
    shap_values = shap.Explanation(
        values = new_phi,
        base_values = np.array([ridge.intercept_]),        # Wartość bazowa – punkt startowy przed dodaniem wpływu cech
        data = new_x_vals,                                      # Oryginalne (niestandaryzowane) wartości – wyświetlane na wykresie
        feature_names = final_names                             # Nazwy kolumn zgodne z wektorem 17 cech
    )

    # Utworzenie tabeli łączącej nazwy kolumn z wartościami od SHAP
    df_shap = pd.DataFrame({
        'features': final_names,                            # Zestaw nazw po grupowaniu
        'shap_values': shap_values.values[0],               # Zapisujemy dane wyliczone metoda SHAP
        'raw_values': shap_values.data[0],                  # Zapisujemy surowe dane
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
                val = row['raw_values']
                # Podmiana na polskie odpowiedniki
                feature_name = polish_feature.get(row['features'], row['features'])
                st.metric(
                    label = feature_name,                   # Nazwa cechy
                    value = f"{float(val):.1f}",                   # Wartość cechy z formularza
                    delta = f"+{row['shap_values']:.2f}"    # Wpływ wartości SHAP w procentach
                )

    with col_minus:
        st.markdown("**📉 Top 3 parametry pogarszające wynik**")
        if df_minus.empty:
            st.info("Brak parametrów o wyraźnie negatywnym wpływie")
        else:
            for _, row in df_minus.iterrows():
                val = row['raw_values']
                # Podmiana na polskie odpowiedniki
                feature_name = polish_feature.get(row['features'], row['features'])
                st.metric(
                    label = feature_name,                   # Nazwa cechy
                    value = f"{float(val):.1f}",                   # Wartość cechy z formularza
                    delta = f"{row['shap_values']:.2f}"     # Wpływ wartości SHAP w procentach
                )

    # Rozwijana sekcja z pełnym wykresem wodospadowym SHAP
    with st.expander("Szczegółowy wykres wodospadowy", expanded = False):
        st.markdown("""
                ### 📖 Jak czytać wykres wodospadowy?
                * **📊 Oś pozioma $f(x)$**: Wynik regeneracji wyrażony w punktach procentowych.
                * **🎯 Wartość bazowa $E[f(x)]$**: średni wynik regeneracji w bazie danych.
                * **🔴 Kolor czerwony**: Paski skierowane w prawo - cechy które **podniosły** wartość.
                * **🔵 Kolor niebieski**: Paski skierowane w lewo - cechy które **obniżyły** wartość.
                * **🏷️ Wartości przy słupkach ($+X.XX$ / $-X.XX$)**: Dokładna liczba punktów dodanych / odjętych przez dany parametr.
                """)

        # Tworzymy nową kartę
        fig, ax = plt.subplots(figsize = (8, 5))
        # Rysujemy wykres wodospadowy
        shap.plots.waterfall(shap_values[0], show = False)
        # Przekazywanie do Streamlit wykres
        st.pyplot(plt.gcf())
        # Czyszczenie pamięci podręcznej
        plt.close()
