FROM python:3.11-slim

WORKDIR /app

COPY db/requirements.txt ./db/requirements.txt
COPY libs/ocr_common ./libs/ocr_common
RUN pip install --no-cache-dir -r db/requirements.txt ./libs/ocr_common

COPY db ./db

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

CMD ["python", "-m", "alembic", "-c", "db/alembic.ini", "upgrade", "head"]
