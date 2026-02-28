# Lightweight Python image for deterministic builds
FROM python:3.10-slim

# Avoid Python output buffering and pyc files in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app source
COPY app.py /app/app.py

# Hugging Face Docker Spaces route traffic through PORT (default 7860)
EXPOSE 7860

# Run Streamlit service on HF-compatible port
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-7860} --server.address 0.0.0.0"]
