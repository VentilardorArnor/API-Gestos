from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from hand_detector import HandDetector
from ai_model import GestureAIModel
import json
import logging
from typing import Optional
import asyncio
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações do servidor
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="API de Reconhecimento de Gestos",
    description="API para reconhecimento de gestos em tempo real usando visão computacional",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa o detector de mãos e o modelo
detector = HandDetector(
    detection_confidence=0.7,
    tracking_confidence=0.7,
    max_hands=1
)

try:
    model = GestureAIModel()
    logger.info("Modelo carregado com sucesso")
except Exception as e:
    logger.error(f"Erro ao carregar o modelo: {str(e)}")
    model = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Nova conexão WebSocket. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Conexão WebSocket fechada. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem: {str(e)}")

manager = ConnectionManager()

@app.get("/")
async def root():
    """Endpoint raiz para verificar se a API está funcionando"""
    return {
        "status": "online",
        "message": "API de Reconhecimento de Gestos está funcionando",
        "version": "1.0.0"
    }

@app.get("/status")
async def get_status():
    """Retorna o status do sistema"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "hand_detector_loaded": detector is not None,
        "active_connections": len(manager.active_connections),
        "server_info": {
            "host": HOST,
            "port": PORT
        }
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket para reconhecimento de gestos em tempo real"""
    await manager.connect(websocket)
    try:
        while True:
            # Recebe o frame codificado em base64
            data = await websocket.receive_text()
            try:
                # Decodifica o frame
                encoded_data = json.loads(data)
                img_data = base64.b64decode(encoded_data["image"])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Detecta mãos no frame
                frame, results = detector.find_hands(frame)
                landmarks = detector.find_positions(frame)

                # Faz a predição se encontrou uma mão
                gesture = None
                if landmarks and model:
                    prediction = model.predict(landmarks)
                    if prediction:
                        gesture = prediction

                # Envia o resultado
                await websocket.send_json({
                    "gesture": gesture,
                    "landmarks": landmarks if landmarks else None
                })

            except Exception as e:
                logger.error(f"Erro ao processar frame: {str(e)}")
                await websocket.send_json({
                    "error": str(e)
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Erro na conexão WebSocket: {str(e)}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=HOST,
        port=PORT,
        reload=False,  # Desativa o reload em produção
        workers=4,     # Número de workers para processamento paralelo
        log_level="info"
    ) 