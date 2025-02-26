import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configurar logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuración del modelo
MODEL_NAME = "distilgpt2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Usando dispositivo: {device}")

# Cargar el tokenizador y el modelo
logger.info(f"Cargando modelo {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

logger.info("Modelo cargado exitosamente.")

# Función para generar respuestas
def generate_response(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200, temperature=0.2, do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
