FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Exposer les ports (API: 8000, Dashboard: 8501)
EXPOSE 8000 8501

# Commande par défaut : lancer l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
