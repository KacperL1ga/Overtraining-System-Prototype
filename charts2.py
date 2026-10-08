import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

def chart_physiology(df: pd.DataFrame):
    if df.empty:
        st.info("Brak danych.")
        return

    df = df.sort_values('date')

    # Tworzymy pustą kartkę/zakladke
    fig_HRV = go.Figure()

    # Naklejami pierwszą warstwę - linia HRV
    # go.Scatter - linia lub punkty (zalezy od parametru "mode")
    fig_HRV.add_trace(
        go.Scatter(
            x = df['date'],             # oś pozioma - data
            y = df['hrv'],              # oś pionowa - wartości HRV
            mode = 'lines + markers',   # 'lines' - linie | 'markers' - punkty
            name = 'HRV',               # etykieta w legendzie
            line = dict(                # kolor i grubosc lini
                color="#1f77b4",
                width=2
            )
        )
    )

    fig_HRV.add_trace(
        go.Scatter(
            x = df['date'],             # oś pozioma - data
            y = df['hrv_baseline'],     # oś pionowa - wartość HRV baseline
            mode = 'lines',             # 'lines' sama linia poglądowa
            name = 'HRV Baseline',      # etykieta w legendzie
            line = dict(
                color="#ff7f0e",
                dash="dash"             # dash tworzy linie przerywaną
            )
        )
    )

    # Opisy na kartce - tytuł i nazwy osi
    fig_HRV.update_layout(
        title = 'HRV vs Baseline w czasie',
        xaxis_title = 'Data',
        yaxis_title = 'HRV (ms)',
        hovermode = "x unified",        # pokazuje wartości obu linie po najechaniu
    )

    # Tworzymy drugą pustą kartkę
    fig_RHR = go.Figure()

    fig_RHR.add_trace(
        go.Scatter(
            x = df['date'],                 # oś pozioma - data
            y = df['resting_heart_rate'],   # oś pionowa - wartości RHR
            name = "RHR",                   # etykieta w legendzie
            mode = 'lines + markers',       # 'lines' - linie | 'markers' - punkty
            line = dict(                    # kolor i grubość lini
                color="#e74c3c",
                width=2
            ),
        )
    )

    fig_RHR.add_trace(
        go.Scatter(
            x = df['date'],                 # oś pozioma - data
            y = df['rhr_baseline'],         # oś pionowa - wartość RHR baseline
            mode = 'lines',                 # 'lines' sama linia poglądowa
            name = "RHR Baseline",          # etykieta w legendzie
            line = dict(
                color="#1f77b4",
                dash='dash'                 # dash - linia przerywana
            )
        )
    )

    fig_RHR.update_layout(
        title = "RHR vs Baseline w czasie",     # Tytuł wykresu
        xaxis_title = "Data",                   # Nazwa lini pionowej
        yaxis_title = "RHR (ms)",               # Nazwa lini poziomej
        hovermode = "x unified",                # Pokazuje wartość obu lini po najechaniu
    )

    # Przekazanie kartki do Streamlita
    st.plotly_chart(fig_HRV, width='stretch')
    st.plotly_chart(fig_RHR, width='stretch')
