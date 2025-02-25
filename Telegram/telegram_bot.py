import os
import sys
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from peft import PeftModel
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuración del modelo
MODEL_DIR = "./deepseek-ciberseguridad-lora"
BASE_MODEL = "CasiAC/deepseek-ciberseguridad-lora"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

# Cargar el tokenizador y el modelo
logger.info("Cargando modelo base...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# Cargar LoRA
logger.info("Aplicando fine-tuning con LoRA...")
model = PeftModel.from_pretrained(model, MODEL_DIR)
model = model.merge_and_unload()
model.to(device)
logger.info("Modelo cargado exitosamente.")

# Función para generar respuestas
def generate_response(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.2, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Handlers de Telegram
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("¡Hola! Soy tu especialista en ciberseguridad. ¿En qué te puedo ayudar?")

async def handle_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    username = update.message.from_user.username
    logger.info(f"Mensaje recibido de {username}: {user_input}")

    response = generate_response(user_input)
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
