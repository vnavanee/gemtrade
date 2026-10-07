# Use a lightweight official Python instance layout
FROM python:3.10-slim

# Set system working folder
WORKDIR /app

# Copy requirement details and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source scripts
COPY main.py .

# Execute the core trading file directly on launch
CMD ["python", "main.py"]
