#ZeroScam

## Instalo librerias
"""
!pip install chromadb fastapi pyngrok uvicorn
!pip install datasets trl
!pip uninstall -y bitsandbytes
!pip install --upgrade bitsandbytes
!pip install --upgrade transformers accelerate
!nvidia-smi
!pip install accelerate
!accelerate config
!pip install flash-attn --no-build-isolation
!pip install pytesseract
!pip install --upgrade python-telegram-bot
!pip install dotenv
!apt-get update
!apt-get install -y tesseract-ocr
!pip install pytesseract
!tesseract -v
import os
import json
import re
import sys
import threading
import logging
import requests
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
from fastapi import FastAPI
import uvicorn
from pyngrok import ngrok
import pytesseract
from PIL import Image, ImageEnhance
import spacy
import nest_asyncio
nest_asyncio.apply()
from huggingface_hub import notebook_login
import torch
import requests
import chromadb
from pyngrok import ngrok
from fastapi import FastAPI
from google.colab import drive
from sentence_transformers import SentenceTransformer
import uvicorn
import threading
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import random
import pytesseract
import spacy
import re
from PIL import Image
from google.colab import files
import sys
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from dotenv import load_dotenv
from PIL import Image, ImageEnhance
from io import BytesIO
"""
# TOKEN ngrok 
ngrok.set_auth_token("Tu Ngrok token")

# VirusTotal API Key
API_KEY = "Tu VirusTotal API Key"

# Telegram Token
TELEGRAM_TOKEN = "Tu telegram bot token"

# CONFIGURACIÓN: Montar Google Drive y Variables
drive.mount('/content/drive')
BASE_DIR = "/content/drive/My Drive/Trabajo Final Bootcamp"
NORMATIVA_DIR = os.path.join(BASE_DIR, "normativa")
EMBEDDINGS_BACKUP_PATH = os.path.join(BASE_DIR, "embeddings.json")

if not os.path.exists(NORMATIVA_DIR):
    raise FileNotFoundError(f"La carpeta {NORMATIVA_DIR} no existe. Verifica la ruta o crea la carpeta.")

# Login en Hugging Face
notebook_login()

# CARGA DE MODELOS Y CONFIGURACIÓN DE DISPOSITIVO
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Usando dispositivo: {device}")

model_name = "CasiAC/deepseek-r1-8b-ciberseguridad"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print("✅ Modelo cargado 🚀")

# CONFIGURACIÓN DE CHROMADB Y FASTAPI
app = FastAPI()

# Configuración de ngrok para exponer el puerto 8000
public_url = ngrok.connect(8000)
print(f"🔗 ChromaDB API accesible en: {public_url}")

client = chromadb.Client()
collection = client.create_collection(
    name="test",
    metadata={"hnsw:search_ef": 100, "hnsw:construction_ef": 1000}
)

@app.get("/")
def read_root():
    return {"message": "Chroma API está en funcionamiento"}

@app.get("/collections")
def get_collections():
    return {"collections": client.list_collections()}

@app.get("/collections/{collection_name}")
def get_collection(collection_name: str):
    return {"collection": client.get_collection(name=collection_name)}

@app.post("/collections/{collection_name}/add")
def add_to_collection(collection_name: str, item: dict):
    coll = client.get_collection(name=collection_name)
    coll.add(
        documents=[item["document"]],
        metadatas=[item.get("metadata", {})],
        ids=[item.get("id", "default_id")]
    )
    return {"message": f"Elemento añadido a la colección {collection_name}"}

@app.get("/health")
def health_check():
    return {"status": "OK"}

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# FUNCIONES DE INDEXACIÓN Y GENERACIÓN DE CONTEXTO (RAG)
def cargar_documentos_y_embeddings():
    embeddings_dict = {}
    for archivo in os.listdir(NORMATIVA_DIR):
        ruta_json = os.path.join(NORMATIVA_DIR, archivo)
        with open(ruta_json, "r", encoding="utf-8") as f:
            documentos = json.load(f)

        for section in documentos.get("sections", []):
            doc_id = f"{archivo}_p{section['page']}"
            content = section["content"]
            embedding = embedding_model.encode(content, convert_to_tensor=True).cpu().numpy().tolist()
            embeddings_dict[doc_id] = embedding

            collection.add(
                documents=[content],
                metadatas=[{"title": documentos.get("title", "Desconocido"), "page": section["page"]}],
                embeddings=[embedding],
                ids=[doc_id]
            )

    with open(EMBEDDINGS_BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(embeddings_dict, f, ensure_ascii=False, indent=4)
    print(f"✅ {len(embeddings_dict)} documentos indexados y guardados en ChromaDB 🚀")
    return embeddings_dict

embeddings_dict = cargar_documentos_y_embeddings()

def obtener_contexto(pregunta, n_docs=3):
    embedding_pregunta = embedding_model.encode([pregunta], convert_to_tensor=True).cpu().numpy().tolist()
    resultados = collection.query(query_embeddings=embedding_pregunta, n_results=n_docs)
    documents = resultados.get('documents', [])
    if not documents:
        return "No se encontraron documentos relevantes."

    documentos_convertidos = [
        " ".join(map(str, doc)) if isinstance(doc, list) else str(doc)
        for doc in documents
    ]
    return "\n".join(documentos_convertidos)

def generar_respuesta_rag(pregunta, max_tokens=300, temperatura=0.1):
    contexto = obtener_contexto(pregunta)
    entrada = f"Contexto: {contexto}\nPregunta: {pregunta}\nRespuesta:"
    inputs = tokenizer(entrada, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperatura,
        top_p=0.9,
        repetition_penalty=1.05
    )
    # Decodificar la salida
    respuesta_completa = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extraer solo la respuesta (eliminar contexto y pregunta)
    respuesta = respuesta_completa.split("Respuesta:")[-1].strip()

    return respuesta

# FUNCIONES DE OCR Y CONSULTA A VIRUSTOTAL
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
nlp = spacy.load("en_core_web_sm")

HEADERS = {"x-apikey": API_KEY}

def preprocesar_imagen(imagen_bytes):
    """Convierte la imagen a escala de grises y mejora el contraste."""
    try:
        if isinstance(imagen_bytes, bytes) and len(imagen_bytes) > 0:  # Aseguramos que recibimos bytes no vacíos
            imagen = Image.open(BytesIO(imagen_bytes))  # Convertimos los bytes en una imagen PIL
        else:
            raise ValueError("Se esperaba un objeto de tipo bytes y no vacío")

        imagen = imagen.convert("L")  # Convertir a escala de grises
        enhancer = ImageEnhance.Contrast(imagen)
        imagen = enhancer.enhance(2.0)  # Aumentar contraste

        return imagen
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

def extraer_texto_img(imagen_bytes):
    """Extrae texto de una imagen tras preprocesarla."""
    try:
        imagen = preprocesar_imagen(imagen_bytes)  # Aquí pasamos los bytes
        if imagen is None:
            return "Error al procesar la imagen"
        # Extraer texto con pytesseract
        texto_extraido = pytesseract.image_to_string(imagen)
        return limpiar_texto(texto_extraido)
    except Exception as e:
        return f"Error al procesar la imagen: {str(e)}"

def consultar_ip(ip):
    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        data = response.json()
        stats = data["data"]["attributes"]["last_analysis_stats"]
        malicious = stats.get("malicious", 0)
        if malicious > 0:
            veredicto = f"❌ La IP {ip} ha sido reportada como maliciosa en {malicious} análisis."
        else:
            veredicto = f"✅ La IP {ip} parece segura."
        return {"IP": ip, "Veredicto": veredicto, "Análisis": stats}
    return {"error": f"Error en la consulta: {response.status_code}"}

def consultar_url(url):
    scan_url = "https://www.virustotal.com/api/v3/urls"
    response = requests.post(scan_url, headers=HEADERS, data={"url": url})
    if response.status_code == 200:
        analysis_id = response.json()["data"]["id"]
        result_url = f"https://www.virustotal.com/api/v3/analyses/{analysis_id}"
        result_response = requests.get(result_url, headers=HEADERS)
        if result_response.status_code == 200:
            data = result_response.json()
            stats = data["data"]["attributes"]["stats"]
            malicious = stats.get("malicious", 0)
            if malicious > 0:
                veredicto = f"❌ La URL {url} ha sido marcada como maliciosa en {malicious} análisis."
            else:
                veredicto = f"✅ La URL {url} parece segura."
            return {"URL": url, "Veredicto": veredicto, "Análisis": stats}
    return {"error": f"Error en la consulta: {response.status_code}"}

def limpiar_texto(texto):
    texto = texto.strip()
    texto = re.sub(r'\s+', ' ', texto)
    return texto.replace("\n", " ").replace("\r", "")

def analizar_con_modelo(texto_extraido):
    """Analiza el mensaje para detectar señales de phishing utilizando el modelo.
       Responde siempre en español.
    """
    texto_limpio = limpiar_texto(texto_extraido)
    system_prompt = """
      Eres un asistente altamente especializado en ciberseguridad. Tu tarea principal es analizar mensajes y detectar intentos de phishing con precisión.
      🔹 REGLAS ESTRICTAS:
      1. No inventes información. Basa tu respuesta ÚNICAMENTE en el texto proporcionado.
      2. Sé conciso y preciso.
      3. La respuesta debe comenzar con "Sospechoso" o "No Sospechoso", seguido de una breve explicación.
      4. Responde siempre en español.
    """
    prompt_modelo = f"""
      Analiza el siguiente mensaje y determina si se trata de un intento de phishing.
      MENSAJE A EVALUAR:
      {texto_extraido}
      Respuesta:
    """
    inputs = tokenizer(system_prompt + prompt_modelo, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=300,
        temperature=0.15,
        do_sample=True,
        top_k=10,
        top_p=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decodificar la salida
    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraer solo la respuesta (eliminar contexto y pregunta)
    respuesta = respuesta_completa.split("Respuesta:")[-1].strip()

    return respuesta


def analizar_prompt(prompt):
    """
    Detecta si el prompt contiene:
      - Una imagen (por extensión) para procesar OCR y análisis de phishing.
      - Una IP o URL para consulta en VirusTotal.
    Devuelve un resultado especial (no None) si se cumple alguno de estos casos.
    """
    if isinstance(prompt, str) and prompt.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.heic')):
        print(f"🔍 Detectada imagen: {prompt}")
        texto_extraido = extraer_texto_img(prompt)
        print(f"Texto extraído: {texto_extraido}")
        return analizar_con_modelo(texto_extraido)

    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    url_pattern = r'(https?://[^\s]+|www\.[^\s]+)'
    if re.search(ip_pattern, prompt):
        ip = re.search(ip_pattern, prompt).group()
        print(f"🔍 Detectada IP: {ip}")
        return consultar_ip(ip)
    if re.search(url_pattern, prompt):
        url = re.search(url_pattern, prompt).group()
        print(f"🔍 Detectada URL: {url}")
        return consultar_url(url)
    return None

# FUNCIÓN UNIFICADA DE GENERACIÓN DE RESPUESTA
def generate_response(prompt):
    """
    Función unificada que:
      1. Verifica si el prompt es especial (imagen, IP, URL) mediante analizar_prompt().
      2. Si no es especial, utiliza la generación basada en contexto (RAG).
      3. Y si no se encuentra relación en el contexto (por ejemplo, se obtiene "No se encontraron documentos relevantes"),
         se usa una generación “por defecto”.
    """
    # Paso 1: Verificar si el prompt es especial
    resultado_api = analizar_prompt(prompt)
    if resultado_api is not None:
        return json.dumps(resultado_api, indent=4, ensure_ascii=False)

    # Paso 2: Consultar contexto en ChromaDB para RAG
    contexto = obtener_contexto(prompt)
    if contexto.strip().lower().startswith("no se encontraron documentos"):
        # Paso 3: Generación por defecto si no hay contexto relevante
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # Si se encontró contexto, usar generación basada en RAG
        return generar_respuesta_rag(prompt)

# CONFIGURACIÓN DEL BOT DE TELEGRAM
# ---------------------------------
load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("¡Hola! Soy tu especialista en ciberseguridad. ¿En qué te puedo ayudar?")

async def handle_message(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    username = update.message.from_user.username
    logger.info(f"Mensaje recibido de {username}: {user_input}")
    response = generate_response(user_input)
    response = response[0:4000]
    await update.message.reply_text(response)

async def handle_photo(update: Update, context: CallbackContext) -> None:
    """Manejador de imágenes enviadas al bot."""
    username = update.message.from_user.username
    photo = update.message.photo[-1]  # Obtiene la mejor calidad de la imagen enviada

    # Descargar la imagen
    photo_file = await photo.get_file()
    img_bytearray = await photo_file.download_as_bytearray()  # Devuelve un bytearray
    img_bytes = bytes(img_bytearray)  # Convertimos bytearray a bytes

    if not img_bytes:
        logger.error("Error: Los bytes de la imagen están vacíos.")
        await update.message.reply_text("No se pudo descargar la imagen.")
        return

    logger.info(f"Imagen recibida de {username}, procesando con OCR...")

    # Extraer texto de la imagen
    texto_extraido = extraer_texto_img(img_bytes)  # Pasamos los bytes correctamente

    if not texto_extraido.strip():
        await update.message.reply_text("No pude extraer texto de la imagen. Asegúrate de que el mensaje sea legible.")
        return

    # Analizar el texto con el modelo
    resultado = analizar_con_modelo(texto_extraido)

    # Responder al usuario con el análisis
    await update.message.reply_text(resultado)

def main():
    if not TELEGRAM_TOKEN:
        logger.error("El token de Telegram no está configurado.")
        sys.exit(1)

    app_telegram = Application.builder().token(TELEGRAM_TOKEN).build()
    app_telegram.add_handler(CommandHandler("start", start))
    app_telegram.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app_telegram.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logger.info("Bot iniciado y ejecutándose...")
    app_telegram.run_polling()

if __name__ == "__main__":
    main()

main()
