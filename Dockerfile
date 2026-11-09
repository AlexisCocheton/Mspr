FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances Python (couche mise en cache tant que requirements.txt ne change pas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source, règles métier et générateur de données
COPY config/ ./config/
COPY reference/generate_dataset.py ./reference/
COPY src/ ./src/

# data/ et models/ ne sont pas dans l'image : ils sont montés en volume
# (docker-compose.yml). Au premier démarrage, le service « trainer » prépare
# les données et entraîne les modèles s'ils sont absents (src/bootstrap.py).

# Ports : API 8000, Dashboard 8501
EXPOSE 8000 8501

# Commande par défaut : lancer l'API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
