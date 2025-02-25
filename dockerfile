# Usar una imagen oficial de Python
FROM python:3.10-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos necesarios
COPY requirements.txt ./
COPY telegram_bot.py ./

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Definir la variable de entorno para el token de Telegram
ENV TELEGRAM_TOKEN="7047664203:AAEa-JEcZQpv-tDCIdV6ZE_odp4lPTH0Bd8"

# Ejecutar el bot cuando el contenedor se inicie
CMD ["python", "telegram_bot.py"]
