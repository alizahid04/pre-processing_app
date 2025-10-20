# Use official Python image (non-slim, more stable for builds)
FROM python:3.11-bullseye

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make sure NLTK uses local nltk_data folder
ENV NLTK_DATA=/app/nltk_data

EXPOSE 5000

CMD ["python", "app_flutter.py"]
