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

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit service
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
