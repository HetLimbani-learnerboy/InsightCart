FROM python:3.11-slim

# --------------------------------------------------
# Environment configuration
# --------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# --------------------------------------------------
# Working directory
# --------------------------------------------------
WORKDIR /app

# --------------------------------------------------
# System dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Python dependencies
# --------------------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# Download NLTK data
# --------------------------------------------------
RUN python -m nltk.downloader \
    stopwords \
    punkt \
    punkt_tab \
    wordnet

# --------------------------------------------------
# Copy project
# --------------------------------------------------
COPY app/ ./app/
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# --------------------------------------------------
# Create log directory
# --------------------------------------------------
RUN mkdir -p logs

# --------------------------------------------------
# FastAPI port
# --------------------------------------------------
EXPOSE 8000

# --------------------------------------------------
# Start application
# --------------------------------------------------
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]