# Overtraining-System-Prototype

Prototyp aplikacji webowej, który na podstawie danych medycznych przesłanych z pliku CSV oraz wyuczonego modelu ML oblicza:
- prawdopodobieństwo wystąpienia przetrenowania,
- poziom regeneracji,
- porady wspierające redukcję ryzyka.

## Uruchomienie

1. Uruchom lokalny serwer HTTP w katalogu repozytorium, np.:
   ```bash
   python -m http.server 8000
   ```
2. Otwórz `http://localhost:8000/index.html` w przeglądarce.
3. Wgraj plik CSV i kliknij **Analizuj**.

## Format danych CSV

Wymagane kolumny:
- `resting_hr`
- `hrv`
- `sleep_hours`
- `training_load`
- `soreness`

Opcjonalna kolumna:
- `mood`

Aplikacja analizuje ostatni wiersz danych w pliku.
