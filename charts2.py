import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

def chart_physiology(df: pd.DataFrame):

    # Jeżeli w tabeli nie ma danych zwracam odpowiedni komunikat
    if df.empty:
        st.info("Brak danych.")
        return

    df = df.sort_values('date')         # Sortujemy dane z bazy chronologicznie po dacie

    # Tworzymy pustą kartkę/zakladke
    fig_HRV = go.Figure()

    # Naklejami pierwszą warstwę - linia HRV
    # go.Scatter - linia lub punkty (zalezy od parametru "mode")
    fig_HRV.add_trace(
        go.Scatter(
            x = df['date'],             # oś pozioma - data
            y = df['hrv'],              # oś pionowa - wartości HRV
            mode = 'lines + markers',   # 'lines' - linie | 'markers' - punkty
            name = 'HRV (ms)',               # etykieta w legendzie
            line = dict(
                color="#1f77b4",        # Kolor lini (błękitny)
                width=2                 # Grubość lini
            )
        )
    )

    fig_HRV.add_trace(
        go.Scatter(
            x = df['date'],             # oś pozioma - data
            y = df['hrv_baseline'],     # oś pionowa - wartość HRV baseline
            mode = 'lines',             # 'lines' sama linia poglądowa
            name = 'HRV Baseline (ms)',      # etykieta w legendzie
            line = dict(
                color="#ff7f0e",        # Kolor lini (pomarańczowy)
                dash="dash"             # dash tworzy linie przerywaną
            )
        )
    )

    # Opisy na kartce - tytuł i nazwy osi
    fig_HRV.update_layout(
        title = 'HRV vs Baseline w czasie', # Tytuł wykresu
        xaxis_title = 'Data',               # Nazwa lini poziomej
        yaxis_title = 'HRV (ms)',           # Nazwa lini pionowej
        hovermode = "x unified",            # pokazuje wartości obu linie po najechaniu
    )

    # Tworzymy drugą pustą kartkę
    fig_RHR = go.Figure()

    fig_RHR.add_trace(
        go.Scatter(
            x = df['date'],                 # oś pozioma - data
            y = df['resting_heart_rate'],   # oś pionowa - wartości RHR
            name = "RHR (bpm)",                   # etykieta w legendzie
            mode = 'lines + markers',       # 'lines' - linie | 'markers' - punkty
            line = dict(
                color="#e74c3c",            # Kolor lini (czerwony)
                width=2                     # Grubość lini
            ),
        )
    )

    fig_RHR.add_trace(
        go.Scatter(
            x = df['date'],                 # oś pozioma - data
            y = df['rhr_baseline'],         # oś pionowa - wartość RHR baseline
            mode = 'lines',                 # 'lines' sama linia poglądowa
            name = "RHR Baseline (bpm)",          # etykieta w legendzie
            line = dict(
                color="#1f77b4",            # Kolor lini (błękitny)
                dash='dash'                 # dash - linia przerywana
            )
        )
    )

    fig_RHR.update_layout(
        title = "RHR vs Baseline w czasie",     # Tytuł wykresu
        xaxis_title = "Data",                   # Nazwa lini pionowej
        yaxis_title = "RHR (bpm)",               # Nazwa lini poziomej
        hovermode = "x unified",                # Pokazuje wartość obu lini po najechaniu
    )

    # Przekazanie kartki do Streamlita
    st.plotly_chart(fig_HRV, width = 'stretch')
    st.plotly_chart(fig_RHR, width = 'stretch')

def  chart_sleep_stacked(df: pd.DataFrame):

    # Jeżeli w tabeli nie ma danych zwracam odpowiedni komunikat
    if df.empty:
        st.info("Brak danych.")
        return

    df = df.sort_values('date')  # Sortujemy dane z bazy chronologicznie po dacie

    # Wyliczamy sen lekki jako wypadkowa snu głebokiego i fazy REM
    df['light_sleep_hours'] = (
            df['sleep_hours'] - df['deep_sleep_hours'] - df['rem_sleep_hours']
    ).clip(lower=0)             # Zabezpieczenie przed wartościami ujemnymi

    # Tworzymy pustą kartkę
    fig = go.Figure()

    # Tworzymy część słupka odpowiedzialną za Sen Głęboki
    fig.add_trace(
        go.Bar(
            x = df['date'],                 # oś pozioma - data
            y = df['deep_sleep_hours'],     # oś pionowa - ilosc godziny snu głebokiego
            name = "Sen głęboki (h)",       # etykieta w legendzie
            marker_color = "#1f77b4",       # kolor słupka (błękitny)
            yaxis = "y1",                   # Przypisanie wartości do lewej osi y
        )
    )

    # Tworzymy część słupka odpowiedzialną za Sen Fazy REM
    fig.add_trace(
        go.Bar(
            x = df['date'],                   # oś pozioma - data
            y = df['rem_sleep_hours'],        # oś pionowa - ilosc godziny snu fazy REM
            name = "Sen REM (h)",             # etykieta w legendzie
            marker_color = "#9467bd",         # kolor słupka (fioletowy)
            yaxis = "y1",                     # Przypisanie wartości do lewej osi y
        )
    )

    # Tworzymy część słupka odpowiedzialną za Sen Lekki
    fig.add_trace(
        go.Bar(
            x = df['date'],                   # oś pozioma - data
            y = df['light_sleep_hours'],      # oś pionowa - ilosc godziny snu lekkiego
            name = "Sen lekki (h)",           # etykieta w legendzie
            marker_color = "#aec7e8",         # kolor słupka (jasny niebieski)
            yaxis = "y1",                     # Przypisanie wartości do lewej osi y
        )
    )

    fig.add_trace(
        go.Scatter(
            x = df['date'],                 # oś pozioma - data
            y = df['sleep_efficiency'],     # oś pionowa - wartości jakości snu
            mode = 'lines + markers',         # lines - linie | markers - punkty
            name = "Efektywność snu (%)",   # etykieta w legendzie
            line = dict(
                color = "#2ecc71",          # Kolor lini (zielony)
                width = 3                   # Grubość lini
            ),
            yaxis = "y2",                   # Przypisujemy wartości do prawej strony
        )
    )

    # Konfiguracja widoku
    fig.update_layout(
        title = "Analiza snu + Efektywność",    # Tytuł wykresu
        xaxis_title = "Data",                   # Nazwa osi poziomej
        barmode = "stack",                      # stack - nakłada słupki na siebie
        yaxis = dict(
            title = "Czas snu (h)",             # Nazwa lewej osi pionowej
            side = "left",                      # Ustawienie po lewej stronie wykresu
        ),
        yaxis2 = dict(
            title = "Efektywność snu (%)",      # Nazwa prawej osi pionowej
            side = "right",                     # Ustawienie po prawej stronie wykresu
            overlaying = "y",                   # Nałożenie osi na ten sam wykres
            range = [0,100],                    # Sztywny zakres 0 - 100
        ),
        hovermode = "x unified"                 # Pokazuje wszystkie wartość po najechaniu na date
    )

    st.plotly_chart(fig, width = 'stretch')

def chart_strain_group (df: pd.DataFrame):
    # Jeżeli w tabeli nie ma danych zwracam odpowiedni komunikat
    if df.empty:
        st.info("Brak danych.")
        return

    df = df.sort_values('date')  # Sortujemy dane z bazy chronologicznie po dacie

    # Tworzymy pustą kartkę
    fig = go.Figure()

    # Tworzymy część słupka odpowiedzialną za Obciążenie Dnia
    fig.add_trace(
        go.Bar(
            x = df['date'],                   # oś pozioma - data
            y = df['day_strain'],             # oś pionowa - obciążenie dnia
            name = "Obciążenie Dnia",              # etykieta w legendzie
            marker_color = "#eb9b54",         # kolor słupka (jasny pomarańczowy)
            yaxis = "y1",                     # Przypisanie wartości do lewej osi y
        )
    )

    # Tworzymy część słupka odpowiedzialną za Obciążenie Treningu
    fig.add_trace(
        go.Bar(
            x = df['date'],                 # oś pozioma - data
            y = df['activity_strain'],      # oś pionowa - ilosc godziny snu lekkiego
            name = "Obciążenie Treningowe",       # etykieta w legendzie
            marker_color="#d35400",         # kolor słupka (ciemny pomarańcz)
            yaxis="y1",                     # Przypisanie wartości do lewej osi y
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df['date'],                   # oś pozioma - data
            y=df['recovery_score'],         # oś pionowa - wartości jakości snu
            mode='lines + markers',         # lines - linie | markers - punkty
            name="Wartość regeneracji (%)", # etykieta w legendzie
            line=dict(
                color="#2ecc71",            # Kolor lini (zielony)
                width=3                     # Grubość lini
            ),
            yaxis="y2",                     # Przypisujemy wartości do prawej strony
        )
    )

    # Konfiguracja widoku
    fig.update_layout(
        title="Analiza wysiłku vs Regeneracja",     # Tytuł wykresu
        xaxis_title="Data",                         # Nazwa osi poziomej
        barmode="group",                            # group - stawia słupki obok siebie
        yaxis=dict(
            title="Obciążenie (0-21)",              # Nazwa lewej osi pionowej
            side="left",                            # Ustawienie po lewej stronie wykresu
            range = [0, 21],                        # Na sztyno ustawione limity
        ),
        yaxis2=dict(
            title="Wskaźnik regeneracji (%)",       # Nazwa prawej osi pionowej
            side="right",                           # Ustawienie po prawej stronie wykresu
            overlaying="y",                         # Nałożenie osi na ten sam wykres
            range=[0, 100],                         # Sztywny zakres 0 - 100
        ),
        hovermode="x unified"                       # Pokazuje wszystkie wartość po najechaniu na date
    )

    st.plotly_chart(fig, width='stretch')