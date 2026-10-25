# Overtraining-System-Prototype
Prototyp systemu służącego do szacowania poziomu regeneracji sportowca na podstawie danych z nocy i dnia poprzedniego. Dodatkowo obliczany jest stosunek przewlekłego i ostrego obciążenia do sprawdzania szans na wystąpienie przetrenowania. Wszystkie obliczenia zebrane są w interaktywne i przejrzyste wykresu dla lepszego zrozumienia trendów, a wszelkie predykcję podparte są tłumaczeniami metody SHAP, wyjaśniającej istotność poszczególnych parametrów na wynik końcowy szacowanych wyników.  

## Uruchomienie aplikacji za pomocą Dockera

Aplikacja została w pełni skonteneryzowana, co oznacza, że **nie musisz instalować środowiska Python ani żadnych bibliotek** na swoim komputerze. Wystarczy zainstalowany [Docker Desktop]

### 1. Wymagania wstępne
* Zainstalowany i uruchomiony program **Docker Desktop**.
* Pobrane pliki projektu (poprzez `git clone` lub pobrany i rozpakowany plik ZIP z repozytorium).

### 2. Krok po kroku

#### Krok 1: Otwórz terminal w katalogu projektu
Przejdź do folderu z pobranym projektem:
```bash
cd sciezka/do/Overtraining-System-Prototype
```

### Krok 2: Zbuduj obraz konterera
Uruchom poniższe polecenie, aby Docker pobrał środowisko i automatycznie skonfigurował aplikację
```bash
docker build -t recovery-app .
```
(Proces ten trwa dłuższą chwilę przy pierwszym uruchomieniu)

### Krok 3: Uruchom kontere
Możesz uruchumić aplikację w trybie standardowym:
```bash
docker run -p 8501:8501 recovery-app
```
### Krok 4: Otwórz aplikację w przeglądarce
Wejdź pod adres: http://localhost:8501

### 3. Zatrzymanie aplikacji
Aby zatrzymać działanie aplikacji, wciśnij w terminalu kombinację klawiszy **CTRL + C** lub zatrzymaj kontener z poziomu interfejsu graficznego Docker Desktop
