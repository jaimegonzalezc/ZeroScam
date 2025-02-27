# Usar una imagen base de Python optimizada para GPU (si usarás CUDA)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 

# Configurar el entorno de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Instalar Python y pip
RUN apt-get install -y python3 python3-pip

# Copiar los archivos del proyecto
COPY . /app

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto 8000 para FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

