# Используем официальный образ PyTorch с поддержкой CUDA 11.8
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Устанавливаем переменные окружения, чтобы apt-get не задавал вопросов
ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update -yqq
RUN apt-get install -yqq --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Открываем порт
EXPOSE 8000

# Команда для запуска сервиса
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]