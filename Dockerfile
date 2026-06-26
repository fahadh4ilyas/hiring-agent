FROM python:3.11-slim

WORKDIR /app

# Install system deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r api/requirements.txt

# Copy application code
COPY . .

# Create directories for runtime artifacts
RUN mkdir -p cache logs

EXPOSE 8000

CMD ["python", "uvicorn.main.py"]
