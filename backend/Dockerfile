FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables
ENV MODEL_PATH="./backend/models"
ENV DATA_PATH="./backend/data"
ENV TEMP_DIR="./backend/temp"
ENV ALLOWED_ORIGINS="*"
ENV MAX_FRAMES="300"
ENV FRAME_SAMPLE_RATE="5"
ENV MIN_DETECTION_CONFIDENCE="0.35"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.gui_ocr:app", "--host", "0.0.0.0", "--port", "8000"]
