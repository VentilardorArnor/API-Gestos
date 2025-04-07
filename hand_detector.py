import mediapipe as mp
import cv2
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Inicializa o detector de mãos usando MediaPipe
        
        Args:
            mode: Se False, trata a imagem como vídeo (mais rápido)
            max_hands: Número máximo de mãos para detectar
            detection_confidence: Confiança mínima para detecção
            tracking_confidence: Confiança mínima para rastreamento
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        """
        Detecta mãos na imagem
        
        Args:
            img: Imagem de entrada (BGR)
            draw: Se True, desenha os pontos de referência na imagem
            
        Returns:
            img: Imagem processada
            results: Resultados da detecção
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
        return img, self.results
    
    def find_positions(self, img, hand_number=0):
        """
        Encontra as posições dos pontos de referência de uma mão específica
        
        Args:
            img: Imagem de entrada
            hand_number: Índice da mão (se houver múltiplas mãos)
            
        Returns:
            landmark_list: Lista de posições dos pontos de referência
        """
        landmark_list = []
        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_number:
                hand = self.results.multi_hand_landmarks[hand_number]
                for id, landmark in enumerate(hand.landmark):
                    height, width, _ = img.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    landmark_list.append([id, cx, cy])
        
        return landmark_list 