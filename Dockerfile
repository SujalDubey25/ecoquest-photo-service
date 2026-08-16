FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# PORT FIX: Render (and most PaaS hosts) inject a $PORT env var and expect
# the container to bind to it — NOT a hardcoded port. Falls back to 7860
# locally so this works identically for local testing and deployment.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
