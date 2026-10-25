# Używamy oficjalnego, lekkiego obrazu Python 3.12
FROM python:3.12-slim

# Ustawienie zmiennych środowiskowych Pythona
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

# Katalog roboczy w kontenerze
WORKDIR /app

# Instalacja podstawowych narzędzi systemowych potrzebnych przy instalacji bibliotek ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Kopiujemy najpierw plik zależności (dla lepszego cachowania warstw Dockera)
COPY requirements.txt .

# Instalacja bibliotek Python
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy resztę plików projektu
COPY . .

# Informujemy Dockera o porcie Streamlita
EXPOSE 8501

# Komenda startowa uruchamiająca aplikację
ENTRYPOINT ["streamlit", "run", "App.py"]
