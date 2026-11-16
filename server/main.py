# EdgeCloudInspection/server/main.py
import base64
import cv2
import numpy as np
import sqlite3
import time
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

app = FastAPI(title="Edge-Cloud Inspection Server")

# 1. Cargar el modelo preentrenado (Se ejecuta al iniciar el contenedor)
MODEL_PATH = "models/mejor_modelo_inspeccion.pt"
model = YOLO(MODEL_PATH)

# 2. Configurar Base de Datos Ligera (SQLite)
DB_FILE = "inspecciones.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS registro
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  clasificacion TEXT,
                  confianza REAL,
                  latencia_ms REAL)''')
    conn.commit()
    conn.close()

init_db()

# 3. Protocolo WebSocket Bidireccional
@app.websocket("/ws/inspeccion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente Edge Conectado.")
    try:
        while True:
            # Recibir frame en formato Base64 desde el Cliente Edge
            data = await websocket.receive_text()
            start_time = time.time()

            # Decodificar Base64 a Matriz OpenCV
            img_data = base64.b64decode(data)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            # Inferencia Custom Vision
            results = model(img, verbose=False)
            probs = results[0].probs
            top1_index = probs.top1
            
            clasificacion = results[0].names[top1_index]
            confianza = float(probs.top1conf)
            
            # Lógica de actuador
            if clasificacion in ["defecto_etiqueta", "defecto_tapa"]:
                accion = "ACTIVAR_BRAZO_RECHAZO"
            else:
                accion = "NINGUNA"

            latencia_ms = round((time.time() - start_time) * 1000, 2)
            timestamp = datetime.now().isoformat()

            # Persistencia de Telemetría
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO registro (timestamp, clasificacion, confianza, latencia_ms) VALUES (?, ?, ?, ?)",
                      (timestamp, clasificacion, confianza, latencia_ms))
            conn.commit()
            conn.close()

            # Estructura de Carga Útil requerida en el enunciado
            respuesta = {
                "timestamp": timestamp,
                "objeto_detectado": clasificacion != "linea_vacia",
                "clasificacion": clasificacion,
                "confianza": round(confianza, 2),
                "accion_actuador": accion,
                "latencia_ms": latencia_ms
            }
            
            # Retornar telemetría al cliente
            await websocket.send_json(respuesta)
            
    except WebSocketDisconnect:
        print("Cliente Edge Desconectado.")

# 4. Telemetría y Trazabilidad (REST)
@app.get("/reporte/turno")
def reporte_turno():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*), clasificacion FROM registro GROUP BY clasificacion")
    filas = c.fetchall()
    conn.close()
    
    total = sum([f[0] for f in filas])
    desglose = {f[1]: f[0] for f in filas}
    
    # Calcular merma (todo lo que no es aprobado ni línea vacía)
    merma = total - desglose.get("aprobado", 0) - desglose.get("linea_vacia", 0)
    porcentaje_defectos = (merma / total * 100) if total > 0 else 0
    
    return {
        "total_inspeccionados": total,
        "porcentaje_defectos": round(porcentaje_defectos, 2),
        "desglose": desglose
    }