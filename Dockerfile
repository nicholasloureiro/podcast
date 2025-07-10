# ------------------------------------------------------------
# Stage 1 – dependências Python
# ------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-deps --wheel-dir /wheels -r requirements.txt

# ------------------------------------------------------------
# Stage 2 – imagem final
# ------------------------------------------------------------
FROM python:3.12-slim

# Evita problemas de fuso/locale
ENV TZ=America/Sao_Paulo \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Instala libs do SO que o ClickHouse-Connect costuma precisar
RUN apt-get update && apt-get install -y --no-install-recommends \
       curl build-essential gcc libc6-dev  \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links /wheels /wheels/*

COPY app/ /app

# Porta default do Streamlit
EXPOSE 8501

# Ajusta a URL base para Streamlit quando em sub-path (opcional)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_RUN_ON_SAVE=true

CMD ["streamlit", "run", "app.py"]
