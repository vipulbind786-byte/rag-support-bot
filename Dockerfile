FROM python:3.11-slim
WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY backend/ backend/
COPY frontend/ frontend/
COPY sample_docs/ sample_docs/

# Deploy config: fastembed (ONNX, no Ollama needed on Spaces) + HF-hosted reranker
ENV EMBED_BACKEND=fastembed
ENV RERANKER_MODEL=BAAI/bge-reranker-base

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
