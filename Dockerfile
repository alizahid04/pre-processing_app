# 1. Base image (Python environment)
FROM python:3.12-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy requirements file first (helps cache dependencies)
COPY requirements.txt .

# 4. Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project
COPY . .

# 6. Expose the port your Flask app uses
EXPOSE 5000

# 7. Command to run the Flask app
CMD ["python", "app_flutter.py"]
