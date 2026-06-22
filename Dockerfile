FROM python:3.12-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN addgroup -g ${GROUP_ID} appgroup && adduser -D -u ${USER_ID} -G appgroup appuser

RUN mkdir -p /app/db /app/logs /app/lock && chown -R appuser:appgroup /app/db /app/logs /app/lock

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser

CMD ["python", "src/scraper.py"]
