import sqlite3

# Nazwa lokalnej bazy danych
DB_NAME = "recovery_app.db"

def init_db():
    # Inicjalizacja bazy danych i tworzenie tabeli 'daily_logs'
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Tworzenie tabeli przechowującej dzienne wpisu
    cursor.execute('''
        CREATE TABLE IF NOT EXIST daily_logs (
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
            wake_ups INTEGER,                   -- Liczba wybudzeń
            
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