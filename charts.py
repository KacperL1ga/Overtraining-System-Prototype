import streamlit as st
import pandas as pd
from datetime import date, timedelta


# ─────────────────────────────────────────────
# SEKCJA 1: FILTROWANIE DANYCH WG DATY
# ─────────────────────────────────────────────

def apply_date_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wyświetla przyciski filtrowania zakresu dat i zwraca przefiltrowany DataFrame.

    session_state zapamiętuje wybrany tryb filtrowania między kliknięciami.
    Bez tego każde przeładowanie strony resetowałoby wybór do domyślnego.
    """

    # Upewniamy się że kolumna 'date' jest obiektem daty, a nie zwykłym tekstem
    df['date'] = pd.to_datetime(df['date'])

    today = pd.Timestamp(date.today())

    st.markdown("**📅 Zakres danych:**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        btn_7      = st.button("Ostatnie 7 dni",  width='stretch')
    with col2:
        btn_30     = st.button("Ostatnie 30 dni", width='stretch')
    with col3:
        btn_all    = st.button("Cały okres",       width='stretch')
    with col4:
        btn_custom = st.button("Własny okres",     width='stretch')

    # Ustawiamy domyślny tryb przy pierwszym otwarciu strony
    if "chart_period" not in st.session_state:
        st.session_state.chart_period = "30"

    # Każdy przycisk nadpisuje tryb w session_state
    if btn_7:
        st.session_state.chart_period = "7"
    if btn_30:
        st.session_state.chart_period = "30"
    if btn_all:
        st.session_state.chart_period = "all"
    if btn_custom:
        st.session_state.chart_period = "custom"

    period = st.session_state.chart_period

    # Na podstawie trybu wyznaczamy datę początkową i końcową
    if period == "7":
        start = today - timedelta(days=7)
        end   = today

    elif period == "30":
        start = today - timedelta(days=30)
        end   = today

    elif period == "all":
        start = df['date'].min()
        end   = df['date'].max()

    else:  # "custom" – ręczny wybór z kalendarza
        c_start, c_end = st.columns(2)
        with c_start:
            # key= musi być unikalny w całej aplikacji, dlatego prefix "chart_"
            start = pd.Timestamp(st.date_input("Data od", value=df['date'].min(), key="chart_start_date"))
        with c_end:
            end   = pd.Timestamp(st.date_input("Data do", value=df['date'].max(), key="chart_end_date"))

    # Filtrujemy wiersze mieszczące się w wybranym przedziale
    mask        = (df['date'] >= start) & (df['date'] <= end)
    df_filtered = df.loc[mask].copy()

    st.caption(f"Widoczne wpisy: {len(df_filtered)} | Zakres: {start.date()} → {end.date()}")

    return df_filtered


# ─────────────────────────────────────────────
# SEKCJA 2: WYKRES – TRENDY FIZJOLOGICZNE vs BASELINE
# ─────────────────────────────────────────────

def chart_physiology(df: pd.DataFrame):
    """
    Wyświetla dwa wykresy liniowe (st.line_chart):
      – HRV rzeczywiste i HRV Baseline
      – RHR rzeczywiste i RHR Baseline

    st.line_chart przyjmuje DataFrame i automatycznie rysuje każdą kolumnę
    jako osobną linię. Wystarczy mu podać tylko te kolumny, które nas interesują.
    """

    if df.empty:
        st.info("Brak danych w wybranym przedziale czasu.")
        return

    # Sortujemy chronologicznie – wykres musi widzieć daty od lewej do prawej
    df = df.sort_values('date').set_index('date')

    st.markdown("#### HRV – zmienność rytmu serca vs Baseline")
    # Wybieramy tylko kolumny potrzebne do tego wykresu
    # Streamlit automatycznie pokoloruje każdą kolumnę na inny kolor
    hrv_cols = ['hrv']
    if 'hrv_baseline' in df.columns:
        hrv_cols.append('hrv_baseline')

    st.line_chart(
        df[hrv_cols].rename(columns={
            'hrv':          'HRV (ms)',
            'hrv_baseline': 'HRV Baseline (ms)'
        })
    )

    st.markdown("#### RHR – tętno spoczynkowe vs Baseline")
    rhr_cols = ['resting_heart_rate']
    if 'rhr_baseline' in df.columns:
        rhr_cols.append('rhr_baseline')

    st.line_chart(
        df[rhr_cols].rename(columns={
            'resting_heart_rate': 'RHR (bpm)',
            'rhr_baseline':       'RHR Baseline (bpm)'
        })
    )


# ─────────────────────────────────────────────
# SEKCJA 3: WYKRES – CZAS I FAZY SNU
# ─────────────────────────────────────────────

def chart_sleep(df: pd.DataFrame):
    """
    Wyświetla dwa wykresy związane ze snem:
      1. st.bar_chart – skumulowane fazy snu (głęboki, REM, lekki)
      2. st.line_chart – efektywność snu w procentach

    Uwaga: natywny st.bar_chart w Streamlicie nie ma trybu "stacked"
    jak Plotly, więc fazy snu wyświetlamy obok siebie w grupach (grouped bars).
    Dla pełnego stacked potrzeba by Plotly lub Altair – tutaj zostawiamy prosto.
    """

    if df.empty:
        st.info("Brak danych w wybranym przedziale czasu.")
        return

    df = df.sort_values('date').set_index('date')

    # Wyliczamy "resztę" snu (lekkiego), zabezpieczając się przed ujemnymi wartościami
    df['light_sleep'] = (
        df['sleep_hours'] - df['deep_sleep_hours'] - df['rem_sleep_hours']
    ).clip(lower=0)

    st.markdown("#### Fazy snu (h)")
    st.bar_chart(
        df[['deep_sleep_hours', 'rem_sleep_hours', 'light_sleep']].rename(columns={
            'deep_sleep_hours': 'Sen głęboki (h)',
            'rem_sleep_hours':  'Sen REM (h)',
            'light_sleep':      'Sen lekki (h)'
        })
    )

    # Efektywność snu jako osobny wykres liniowy (inny zakres wartości: 0-100%)
    if 'sleep_efficiency' in df.columns and df['sleep_efficiency'].notna().any():
        st.markdown("#### Efektywność snu (%)")
        st.line_chart(
            df[['sleep_efficiency']].rename(columns={
                'sleep_efficiency': 'Efektywność snu (%)'
            })
        )


# ─────────────────────────────────────────────
# SEKCJA 4: WYKRES – OBCIĄŻENIE vs REGENERACJA
# ─────────────────────────────────────────────

def chart_strain_vs_recovery(df: pd.DataFrame):
    """
    Zestawia dwa wykresy ilustrujące zależność między wysiłkiem a regeneracją:
      1. st.bar_chart – Day Strain i Activity Strain (jak ciężki był dzień)
      2. st.line_chart – Recovery Score (jak dobrze się zregenerowaliśmy)

    Wykresy są osobne, bo mają różne skale (Strain: 0-21, Recovery: 0-100%).
    Użytkownik wizualnie porównuje, co dzieje się z regeneracją dzień po ciężkim treningu.
    """

    if df.empty:
        st.info("Brak danych w wybranym przedziale czasu.")
        return

    df = df.sort_values('date').set_index('date')

    st.markdown("#### Obciążenie dnia (Day Strain i Activity Strain)")
    strain_cols = ['day_strain']
    if 'activity_strain' in df.columns and df['activity_strain'].notna().any():
        strain_cols.append('activity_strain')

    st.bar_chart(
        df[strain_cols].rename(columns={
            'day_strain':      'Day Strain (0–21)',
            'activity_strain': 'Activity Strain (0–21)'
        })
    )

    # Recovery Score osobno jako linia – inna skala (%)
    if 'recovery_score' in df.columns and df['recovery_score'].notna().any():
        st.markdown("#### Recovery Score (%)")
        st.line_chart(
            df[['recovery_score']].rename(columns={
                'recovery_score': 'Recovery Score (%)'
            })
        )


# ─────────────────────────────────────────────
# SEKCJA 5: GŁÓWNA FUNKCJA EKSPORTOWANA DO App.py
# ─────────────────────────────────────────────

def render_charts_section(user_id: str, df_history: pd.DataFrame):
    """
    Jedyna funkcja, którą importujesz i wywołujesz z App.py:

        from charts import render_charts_section
        render_charts_section(user_id, get_user_history(user_id))

    Kolejność działania:
      1. Sprawdza czy są dane do wyświetlenia
      2. Filtruje historię wg wybranego zakresu dat
      3. Dzieli widok na 3 zakładki i wywołuje odpowiednie funkcje rysujące
    """

    st.subheader("📈 Analiza Trendów")

    if df_history.empty:
        st.info("Brak danych historycznych. Dodaj co najmniej kilka wpisów aby zobaczyć wykresy.")
        return

    # Filtrujemy dane – apply_date_filter() wyświetla przyciski i zwraca odcięty DataFrame
    df_filtered = apply_date_filter(df_history)

    if df_filtered.empty:
        st.warning("Brak wpisów w wybranym zakresie dat. Rozszerz filtr.")
        return

    # Tworzymy 3 zakładki – kliknięcie przełącza widok bez przeładowania strony
    tab1, tab2, tab3 = st.tabs([
        "🫀 Fizjologia vs Baseline",
        "🛏️ Fazy Snu",
        "💪 Obciążenie vs Regeneracja"
    ])

    with tab1:
        chart_physiology(df_filtered)

    with tab2:
        chart_sleep(df_filtered)

    with tab3:
        chart_strain_vs_recovery(df_filtered)
