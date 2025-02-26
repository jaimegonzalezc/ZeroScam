import os
import sys
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv
from model_loader import generate_response  # Importa la función desde model_loader.py

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Handlers de Telegram
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("¡Hola! Soy tu especialista en ciberseguridad. ¿En qué te puedo ayudar?")

async def handle_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    username = update.message.from_user.username
    logger.info(f"Mensaje recibido de {username}: {user_input}")

    response = generate_response(user_input)  # Usa la función importada
    await update.message.reply_text(response)

# Función principal
def main():
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    if not TELEGRAM_TOKEN:
        logger.error("El token de Telegram no está configurado.")
        sys.exit(1)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot iniciado y ejecutándose...")
    app.run_polling()

if __name__ == "__main__":
    main()
