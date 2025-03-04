# ZeroScam

# RAG Normativa de Ciberseguridad

La técnica **Retrieval-Augmented Generation (RAG)**, o **Generación Aumentada por Recuperación**, 
es una metodología innovadora que combina la potencia de los modelos de lenguaje de gran escala (LLM)
con la capacidad de recuperar información externa y actualizada. Esto permite que el modelo no se limite
únicamente a su conocimiento preentrenado, sino que pueda acceder a datos recientes y específicos, lo 
que resulta fundamental en un campo tan dinámico como la ciberseguridad.

---

## ¿Qué es RAG?

RAG integra dos componentes esenciales:

1. **Recuperación (Retrieval):**  
   Permite buscar y extraer información relevante de bases de datos o documentos normativos, brindando 
   al modelo acceso a datos actualizados y especializados.

2. **Generación (Generation):**  
   Utiliza un modelo de lenguaje para generar respuestas enriquecidas con la información recuperada. 
   En este caso, se utiliza el modelo **Deepseek R1 8B Ciberseguridad**, que ha sido previamente entrenado
   mediante técnicas de **Fine-tuning** y **LoRA**.

Esta combinación mejora notablemente la precisión y relevancia de las respuestas, ya que el modelo puede 
complementar su conocimiento interno con datos externos verificados.

---

## Aplicación en la Normativa de Ciberseguridad

El notebook implementa una solución RAG para potenciar el rendimiento del modelo **Deepseek R1 8B Ciberseguridad**. 
La integración de la normativa de ciberseguridad permite que el modelo:

- **Acceda a información actualizada:**  
  Consulta en tiempo real documentos normativos y estándares reconocidos.

- **Genere respuestas contextualizadas:**  
  Combina su conocimiento preentrenado con información específica y reciente para ofrecer respuestas fundamentadas.

- **Mejore la precisión:**  
  Al basar sus respuestas en datos oficiales, aumenta la fiabilidad y relevancia de la información proporcionada.

---

## Documentos Normativos Utilizados

El sistema RAG se apoya en un conjunto de documentos clave que abarcan diversos aspectos de la ciberseguridad:

- **GDPR 2016/679:**  
  Reglamento General de Protección de Datos de la Unión Europea, que establece las bases para la protección 
  de datos personales y la privacidad.

- **GDPR 2018/1725:**  
  Normativa orientada a la protección de datos en instituciones de la Unión Europea, aplicada en organismos 
  públicos y entidades gubernamentales.

- **ISO 27001:2022:**  
  Norma internacional para la gestión de la seguridad de la información, que proporciona un marco para establecer,
  implementar y mejorar un Sistema de Gestión de Seguridad de la Información (SGSI).

- **NIST:**  
  Marco de seguridad cibernética del Instituto Nacional de Estándares y Tecnología (EE.UU.), que ofrece directrices, 
  mejores prácticas y estándares para la gestión del riesgo y la protección de la información.

---

## Arquitectura del Sistema

El sistema se compone de varias partes integradas que permiten su funcionamiento de manera conjunta:

- **Recuperación de Documentos:**  
  Los documentos normativos se indexan y almacenan en una base de datos (por ejemplo, ChromaDB) para facilitar su 
  búsqueda y recuperación.

- **Generación Aumentada (RAG):**  
  El modelo Deepseek utiliza la información recuperada para generar respuestas detalladas y precisas, aprovechando
  tanto su conocimiento preentrenado como los datos externos.

- **Interfaz de API:**  
  Se emplea **FastAPI** para exponer el servicio, y mediante **ngrok** se genera una URL pública que permite 
  integrar y probar la solución en entornos remotos, como Google Colab.

---

## Beneficios de la Solución RAG en Ciberseguridad

La integración de la normativa en un sistema RAG ofrece múltiples ventajas:

- **Actualización Continua:**  
  El modelo puede acceder a información normativa actualizada, lo que es esencial en un campo en constante evolución.

- **Respuestas Contextualizadas y Precisasy:**  
  Las respuestas se enriquecen con datos específicos y relevantes, lo que mejora la calidad y exactitud de las 
  soluciones proporcionadas.

- **Facilidad de Integración:**  
  La combinación de FastAPI y ngrok permite exponer el servicio a entornos remotos, facilitando la integración con 
  otras aplicaciones y sistemas.

- **Adaptabilidad y Escalabilidad:**  
  La técnica RAG es flexible y puede ampliarse para cubrir otros ámbitos o conjuntos de datos, aumentando así su 
  utilidad en diversas áreas del conocimiento.

---

## Conclusión

Este notebook demuestra cómo la técnica **RAG (Retrieval-Augmented Generation)** puede revolucionar la forma de 
abordar la ciberseguridad mediante:

- **Recuperación de Información Actualizada:**  
  El modelo accede a documentos normativos y datos recientes para fundamentar sus respuestas.

- **Generación de Respuestas Contextualizadas:**  
  Al combinar su conocimiento interno con información externa, el modelo ofrece respuestas más precisas y adaptadas
  a las necesidades reales del entorno de la seguridad de la información.

- **Integración Sencilla y Eficiente:**  
  Gracias a FastAPI y ngrok, el sistema se integra fácilmente en entornos remotos, permitiendo pruebas y adaptaciones 
  rápidas.

Con esta configuración, el modelo **Deepseek R1 8B Ciberseguridad** puede proporcionar respuestas informadas, 
relevantes y contextualizadas a preguntas relacionadas con la seguridad de la información, marcando un avance 
significativo en la aplicación de técnicas RAG en el ámbito de la ciberseguridad.
