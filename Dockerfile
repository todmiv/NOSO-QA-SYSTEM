FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh
RUN ollama pull qwen2.5:3b

# Copy project files
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Create dirs
RUN mkdir -p data/chroma_db

CMD ["python", "rag_app.py"]
