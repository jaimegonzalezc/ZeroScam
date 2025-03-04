<center><h1>ZeroScam</h1></center>

## Tesseract OCR

### Introducción

Este módulo implementa un sistema de reconocimiento óptico de caracteres (OCR) optimizado, utilizando Tesseract como motor principal. Se realizaron pruebas comparativas con diferentes frameworks de OCR para determinar la mejor opción en términos de precisión y rendimiento.

### Evaluación de Modelos OCR

Se analizaron tres soluciones de OCR para determinar cuál ofrecía el mejor desempeño en términos de exactitud y velocidad:

- **PaddleOCR**
- **EasyOCR**
- **Tesseract**

Tras una serie de pruebas, **Tesseract fue seleccionado** como la mejor opción debido a su equilibrio entre precisión y eficiencia en el procesamiento de textos en imágenes.

### Uso

Para procesar una imagen y extraer su texto, se emplea la función `extraer_texto_img`, que incluye técnicas de preprocesamiento antes de aplicar Tesseract OCR. La implementación utilizada en el notebook es la siguiente:

```python
from PIL import Image, ImageEnhance
import pytesseract

def limpiar_texto(texto):
    """ Elimina espacios y carácteres no deseados. """
    texto = texto.strip()
    texto = re.sub(r'\s+', ' ', texto)
    return texto.replace("\n", " ").replace("\r", "")

def preprocesar_imagen(imagen):
    """Convierte la imagen a escala de grises y mejora el contraste."""
    try:
        image = Image.open(imagen).convert("L")
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(2.0)
    except Exception as e:
        return None

def extraer_texto_img(imagen):
    """Extrae y limpia el texto de una imagen."""
    image = preprocesar_imagen(imagen)
    if image is None:
        return "Error al procesar la imagen"
    texto_extraido = pytesseract.image_to_string(image)
    return limpiar_texto(texto_extraido)


```

### Detalles Técnicos

Para mejorar la precisión del OCR, se aplicaron las siguientes técnicas de preprocesamiento:

- **Conversión a escala de grises**: Reduce el ruido de color y mejora la detección de caracteres.
- **Ajuste de contraste**: Mejora la legibilidad del texto en imágenes con iluminación deficiente.

Estas optimizaciones garantizan que Tesseract reciba imágenes en un formato óptimo para la detección de texto.

### Resultados

Ejemplo de salida esperada:

```
Texto extraído de la imagen con OCR optimizado.
```

