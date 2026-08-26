# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies
# ffmpeg is needed for many audio processing tasks
# libsqlite3-dev is needed for the SQLite database
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model so cold starts are fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . .

# Expose the standard Hugging Face Space port
EXPOSE 7860

# Command to run the FastAPI app with Uvicorn targeting port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
