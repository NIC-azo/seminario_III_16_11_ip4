# cliente_edge.py
import cv2
import json
import base64
import asyncio
import websockets
from ultralytics import YOLO

# 1. Cargar modelo YOLOv8 nano preentrenado (descarga automática la primera vez)
# Este modelo solo se usará para encontrar DÓNDE está la botella (Object Detection)
modelo_deteccion = YOLO("yolov8n.pt")

# IP del servidor (Si pruebas en la misma PC usa localhost o 127.0.0.1)
# Si tu compañero prueba desde otra PC, pon la IP local de TU PC (ej. 192.168.1.50)
URI_WEBSOCKET = "ws://127.0.0.1:8000/ws/inspeccion"

async def procesar_video():
    cap = cv2.VideoCapture(0)  # 0 es la webcam por defecto
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    # Conectar al WebSocket del servidor
    async with websockets.connect(URI_WEBSOCKET) as websocket:
        print("✅ Conectado al Servidor Edge-Cloud")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Detección Previa (Encontrar el objeto en la faja)
            resultados = modelo_deteccion(frame, verbose=False)
            boxes = resultados[0].boxes
            
            # Variables HUD por defecto
            estado_texto = "Esperando producto..."
            color_caja = (200, 200, 200) # Gris
            latencia = 0
            
            # Si se detecta al menos un objeto
            if len(boxes) > 0:
                # Tomar el primer objeto detectado con mayor confianza
                x1, y1, x2, y2 = map(int, boxes[0].xyxy[0])
                
                # 3. Recorte de la Región de Interés (ROI)
                roi = frame[y1:y2, x1:x2]
                
                # 4. Codificar ROI a Base64 para enviar por red local
                _, buffer = cv2.imencode('.jpg', roi)
                imagen_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Enviar frame al servidor y esperar respuesta
                await websocket.send(imagen_b64)
                respuesta_str = await websocket.recv()
                
                # Procesar JSON del servidor
                datos = json.loads(respuesta_str)
                clasificacion = datos.get("clasificacion", "desconocido")
                confianza = datos.get("confianza", 0)
                latencia = datos.get("latencia_ms", 0)
                accion = datos.get("accion_actuador", "")

                # 5. Renderizado del Dashboard HUD (Lógica visual)
                if clasificacion == "aprobado":
                    color_caja = (0, 255, 0) # Verde
                    estado_texto = f"APROBADO ({confianza:.2f})"
                elif clasificacion in ["defecto_etiqueta", "defecto_tapa"]:
                    color_caja = (0, 0, 255) # Rojo
                    estado_texto = f"RECHAZADO: {clasificacion}"
                
                # Dibujar bounding box y etiquetas
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_caja, 2)
                cv2.putText(frame, estado_texto, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_caja, 2)
                
                # Mostrar Latencia en la esquina
                cv2.putText(frame, f"Latencia: {latencia} ms", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Alerta visual si se activa actuador
                if accion == "ACTIVAR_BRAZO_RECHAZO":
                    cv2.putText(frame, "!!! ALERTA DEFECTO !!!", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Mostrar video en vivo
            cv2.imshow('HUD - Nodo Edge Inspection', frame)

            # Presionar 'q' para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Iniciar ciclo asíncrono
    asyncio.run(procesar_video())