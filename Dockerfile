# Mini Local Analytics Assistant - local container image.
#
# 3.13 matches the interpreter the project is developed and tested against, so
# the pinned pandas/matplotlib/duckdb wheels resolve identically here.
FROM python:3.13-slim

# Unbuffered output so logs appear immediately in `docker compose logs`;
# no .pyc files, since the image layer is read-only in practice.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Requirements first: this layer is cached and only rebuilt when they change.
# No system packages are installed - duckdb, pandas and matplotlib all ship
# manylinux wheels, so a compiler toolchain is not needed.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source only. .dockerignore keeps .env, databases, charts, logs,
# caches and the virtualenv out of the build context entirely.
COPY *.py ./
COPY templates/ ./templates/
COPY static/ ./static/
COPY data/ ./data/

# Directories the app writes to. They are normally bind-mounted by Compose;
# creating them here means the image also runs standalone without mounts.
# Run as a non-root user, owning only what it needs to write.
RUN mkdir -p /app/charts /app/logs /app/docker-data \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Verifies the app answers HTTP. /api/status touches no database and makes no
# Groq call, so the healthcheck consumes no quota. Uses the standard library,
# so curl does not need installing.
HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=5 \
    CMD ["python", "-c", "import os,sys,urllib.request; port=os.getenv('WEB_PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/api/status', timeout=4).status==200 else 1)"]

CMD ["python", "-u", "web_app.py"]
