# Imagen base con Python y soporte para CUDA 12.1
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Establece el directorio de trabajo
WORKDIR /app

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv git wget curl \
    && rm -rf /var/lib/apt/lists/*

# Crea un entorno virtual y actívalo
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia el archivo de dependencias
COPY requirements.txt .

# Instala las librerías necesarias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copia solo la carpeta del bot al contenedor
COPY Telegram/ /app/Telegram/

# Establece la carpeta de trabajo dentro del contenedor
WORKDIR /app/Telegram

# Expone el puerto para el bot de Telegram
EXPOSE 8080

# Comando para ejecutar el bot
CMD ["python", "bot.py"]