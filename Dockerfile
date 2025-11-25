FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models data

ENV WANDB_API_KEY=${WANDB_API_KEY}

EXPOSE 8080

# Default to FastAPI serving
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]