import sqlite3

import pandas as pd

# Nazwa lokalnej bazy danych
DB_NAME = "recovery_app.db"

def init_db():
    # Inicjalizacja bazy danych i tworzenie tabeli 'daily_logs'
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Tworzenie tabeli przechowującej dzienne wpisu
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,              -- Identyfikator uzytkownika
            date DATE NOT NULL,                 -- Data pomiaru (YYYY-MM-DD)
            
            hrv REAL,                           -- Zmiennosc rytmu serca (ms)
            resting_heart_rate REAL,            -- Tętno spoczynkowe (bpm)
            skin_temp_deviation REAL,           -- Odchylenie temperatury działa (= x - 36.6)
            
            hrv_baseline REAL,                  -- Wartość bazowa HRV
            rhr_baseline REAL,                  -- Wartość bazowa RHR
            hrv_deviation REAL,                 -- Odchylenie HRV od baseline'u
            rhr_deviation REAL,                 -- Odchylenie RHR od baseline'u
            
            sleep_hours REAL,                   -- Łączny czas snu (h)
            sleep_efficiency REAL,              -- Efentywność snu (%)
            deep_sleep_hours REAL,              -- Sen głęboki (%)
            rem_sleep_hours REAL,               -- Sen REM (%)
            
            day_strain REAL,                    -- Skala obciążenia (0-21)
            calories_burned REAL,               -- Całkowite spalone kalorie (kcal)
            workout_completed INTEGER,          -- Trening wykonany (1-TAK 0-NIE)
            
            recovery_score REAL,                -- Wynik regeneracji (0-100)
            
            UNIQUE(user_id, date)               -- Jeden zapis pomiarów w ciągu dnia
        )
    ''')

    # Zapisanie zmian w bazie i zamknięcie połączenia
    conn.commit()
    conn.close()

def save_daily_logs(data: dict):
    # Zapisywanie nowych wartości lub aktualizowanie istniejących
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Komenda SQL do zmian w wartosciach, dzieki kluczom ze slowinika
    sql = '''
        INSERT OR REPLACE INTO daily_logs (
            user_id, date,
            hrv, resting_heart_rate, skin_temp_deviation,
            hrv_baseline, rhr_baseline, hrv_deviation, rhr_deviation,
            sleep_hours, sleep_efficiency, deep_sleep_hours, rem_sleep_hours,
            day_strain, calories_burned, workout_completed,
            recovery_score
        ) VALUES (
            :user_id, :date,
            :hrv, :resting_heart_rate, :skin_temp_deviation,
            :hrv_baseline, :rhr_baseline, :hrv_deviation, :rhr_deviation,
            :sleep_hours, :sleep_efficiency, :deep_sleep_hours, :rem_sleep_hours,
            :day_strain, :calories_burned, :workout_completed,
            :recovery_score
        )
    '''
    cursor.execute(sql, data)
    conn.commit()
    conn.close()

def get_user_history(user_id: str) -> pd.DataFrame:
    # Pobieranie historii danych uzytkownika z Bazy Danych w postaci DataFrame
    conn = sqlite3.connect(DB_NAME)

    # Komenda zwracająca dane konkretnego uzytkownika, posortowane po dacie
    query = "SELECT * FROM daily_logs WHERE user_id = ? ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=(user_id,))

    conn.close()
    return df

def delete_daily_log(user_id: str, date_str: str):
    # Usuwanie konkretnego wpisu z bazy danych
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    sql = "DELETE FROM daily_logs WHERE user_id = ? AND date = ?"
    cursor.execute(sql, (user_id, date_str))
    
    conn.commit()
    conn.close()
