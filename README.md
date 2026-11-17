# Mini-Azure Vision: Sistema Local de Control de Uniforme On-Premise

Sistema de control de ingreso automatizado mediante visión artificial para verificar en tiempo real el uso correcto del uniforme (corbata) en estudiantes. Este proyecto implementa una arquitectura On-Premise (Nube Privada) utilizando microservicios locales para garantizar baja latencia, privacidad de datos y optimización de costos frente a soluciones de nube pública.

## 🏗️ Arquitectura y Tecnologías

El sistema está dividido en una arquitectura Cliente-Servidor comunicada a través de la red de área local (LAN).

- **IA y Modelos:** Ultralytics YOLOv8 (modelo base `yolov8n.pt` para detección de presencia y modelo _fine-tuned_ `best.pt` para clasificación del uniforme).
- **Backend (Servidor):** FastAPI, Uvicorn, Python 3.10.
- **Despliegue:** Docker, Docker Compose.
- **Cliente (Streaming):** OpenCV, Requests (Transmisión de fotogramas en Base64 mediante peticiones HTTP POST).

---

## 🚀 Instalación y Despliegue

### 1. Servidor API REST (Integrante A - Rol Cloud/IA)

El servidor expone los servicios de inferencia empaquetados en un contenedor Docker.

**Requisitos previos:** Docker y Docker Compose instalados.

1. Clonar el repositorio y navegar a la carpeta raíz del proyecto.
2. Asegurarse de que el modelo entrenado (`best.pt`) esté ubicado en `server/model/best.pt`.
3. Configurar las variables en `server/.env`:
   ```env
   BACKEND_PORT=8000
   CONTAINER_NAME=mini_azure_server
   MODEL_PATH=model/best.pt
   CONFIDENCE_THRESHOLD=0.80
   ```
