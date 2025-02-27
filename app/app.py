## Código sin FastAPI

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import requests
import re

# Cargar modelo entrenadoimport torch
# Definir el dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Nombre del modelo en Hugging Face
model_name = "CasiAC/deepseek-r1-8b-ciberseguridad"

# Configuración de quantización en 4 bits para optimizar memoria
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Carga el modelo en 4 bits en lugar de 8 bits
    bnb_4bit_compute_dtype=torch.float16,  # Usa FP16 para cálculos
    bnb_4bit_use_double_quant=True,  # Habilita doble cuantización para optimización
    bnb_4bit_quant_type="nf4"  # Cuantización NF4 (recomendada por Hugging Face)
)

# Cargar el modelo con cuantización en 4 bits
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Distribuye el modelo entre CPU/GPU automáticamente
    trust_remote_code=True
)

# Cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 🔹 API Key de VirusTotal
API_KEY = "06858db9f480b4aba21a5831457a9b919b1f9014e6f8872ee1f4f7d1a029197c"
HEADERS = {"x-apikey": API_KEY}

def consultar_ip(ip):
    """Consulta una IP en VirusTotal y evalúa si es segura o maliciosa."""
    url = f"https://www.virustotal.com/api/v3/ip_addresses/{ip}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        data = response.json()
        stats = data["data"]["attributes"]["last_analysis_stats"]
        malicious = stats.get("malicious", 0)
        harmless = stats.get("harmless", 0)

        if malicious > 0:
            veredicto = f"❌ La IP {ip} ha sido reportada como **maliciosa** en {malicious} análisis."
        else:
            veredicto = f"✅ La IP {ip} parece **segura**, sin reportes de actividad maliciosa."

        return {
            "IP": ip,
            "Veredicto": veredicto,
            "Análisis": stats
        }
    return {"error": f"Error en la consulta: {response.status_code}"}

def consultar_url(url):
    """Consulta una URL en VirusTotal y evalúa si es segura o maliciosa."""
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
            harmless = stats.get("harmless", 0)

            if malicious > 0:
                veredicto = f"❌ La URL {url} ha sido **marcada como maliciosa** en {malicious} análisis."
            else:
                veredicto = f"✅ La URL {url} parece **segura**, sin reportes de actividad maliciosa."

            return {
                "URL": url,
                "Veredicto": veredicto,
                "Análisis": stats
            }

    return {"error": f"Error en la consulta: {response.status_code}"}

def analizar_prompt(prompt):
    """Detecta si el prompt contiene una IP o URL y consulta VirusTotal si es necesario."""
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    url_pattern = r"https?://[^\s/$.?#].[^\s]*"

    ip_match = re.search(ip_pattern, prompt)
    url_match = re.search(url_pattern, prompt)

    if ip_match:
        ip = ip_match.group()
        print(f"🔍 Detectada IP en el prompt: {ip}")
        return consultar_ip(ip)

    if url_match:
        url = url_match.group()
        print(f"🔍 Detectada URL en el prompt: {url}")
        return consultar_url(url)

    return None  # No se detectó ninguna IP o URL

def generar_respuesta(prompt):
    """Genera una respuesta con el modelo o consulta VirusTotal si es necesario."""
    resultado_api = analizar_prompt(prompt)

    if resultado_api:
        return json.dumps(resultado_api, indent=4, ensure_ascii=False)  # Respuesta en JSON

    # Tokenizar el prompt y generar la respuesta
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decodificar y limpiar la salida
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return response.replace(prompt, "").strip()

# 🔹 Ejemplo de uso con una IP y una URL
prompt_usuario_1 = "¿Esta IP 8.8.8.8 es segura?"
prompt_usuario_2 = "¿La URL http://malicious-site.com es peligrosa?"
prompt_usuario_3 = "¿Cómo puedo protegerme del phishing?"

respuesta_1 = generar_respuesta(prompt_usuario_1)
respuesta_2 = generar_respuesta(prompt_usuario_2)
respuesta_3 = generar_respuesta(prompt_usuario_3)

print(f"\n🔹 Respuesta para IP:\n{respuesta_1}")
print(f"\n🔹 Respuesta para URL:\n{respuesta_2}")
print(f"\n🔹 Respuesta normal del modelo:\n{respuesta_3}")