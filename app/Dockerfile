# Usa una imagen base con Python y CUDA preinstalado
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Establece el directorio de trabajo
WORKDIR /app

# Actualiza e instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos de la aplicación al contenedor
COPY . /app

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Expone los puertos necesarios
EXPOSE 8000 
EXPOSE 8443 

# Comando de ejecución
CMD ["python3", "zeroscam.py"]
