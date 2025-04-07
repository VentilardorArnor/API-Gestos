import numpy as np
from enum import Enum

class Gesture(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"  # Mão aberta
    PEACE = "peace"          # Paz e amor
    POINTING = "pointing"    # Dedo indicador
    CLOSED_FIST = "fist"    # Mão fechada
    PINCH = "pinch"         # Pinça (polegar e indicador)

class GestureRecognizer:
    def __init__(self):
        """
        Inicializa o reconhecedor de gestos
        """
        # Índices dos dedos (começo e ponta)
        self.finger_indices = {
            'thumb': [4, 3, 2, 1],    # Polegar
            'index': [8, 7, 6, 5],    # Indicador
            'middle': [12, 11, 10, 9], # Médio
            'ring': [16, 15, 14, 13],  # Anelar
            'pinky': [20, 19, 18, 17]  # Mindinho
        }

    def _is_finger_extended(self, landmarks, finger_indices):
        """
        Verifica se um dedo está estendido baseado em seus pontos de referência
        """
        if not landmarks:
            return False

        tip = landmarks[finger_indices[0]]
        pip = landmarks[finger_indices[2]]
        
        return tip[2] < pip[2]  # Compara as coordenadas Y

    def _is_thumb_extended(self, landmarks, finger_indices):
        """
        Verifica se o polegar está estendido (lógica especial devido à sua orientação)
        """
        if not landmarks:
            return False

        tip = landmarks[finger_indices[0]]
        pip = landmarks[finger_indices[2]]
        
        return tip[1] > pip[1]  # Compara as coordenadas X

    def _is_pinch(self, landmarks):
        """
        Verifica se há um gesto de pinça entre o polegar e o indicador
        """
        if not landmarks or len(landmarks) < 21:
            return False

        # Pontos de referência do polegar e indicador
        thumb_tip = landmarks[4]  # Ponta do polegar
        index_tip = landmarks[8]  # Ponta do indicador

        # Calcula a distância entre as pontas
        distance = np.sqrt((thumb_tip[1] - index_tip[1])**2 + (thumb_tip[2] - index_tip[2])**2)
        
        # Se a distância for menor que 30 pixels, considera como pinça
        return distance < 30

    def recognize_gesture(self, landmarks):
        """
        Reconhece o gesto baseado nos pontos de referência da mão
        
        Args:
            landmarks: Lista de pontos de referência da mão
            
        Returns:
            Gesture: Gesto reconhecido
        """
        if not landmarks or len(landmarks) < 21:
            return Gesture.NONE

        # Verifica cada dedo
        thumb_extended = self._is_thumb_extended(landmarks, self.finger_indices['thumb'])
        index_extended = self._is_finger_extended(landmarks, self.finger_indices['index'])
        middle_extended = self._is_finger_extended(landmarks, self.finger_indices['middle'])
        ring_extended = self._is_finger_extended(landmarks, self.finger_indices['ring'])
        pinky_extended = self._is_finger_extended(landmarks, self.finger_indices['pinky'])

        # Verifica primeiro o gesto de pinça
        if self._is_pinch(landmarks):
            return Gesture.PINCH

        # Reconhecimento de outros gestos
        if all([index_extended, middle_extended, ring_extended, pinky_extended]):
            return Gesture.OPEN_PALM  # Mão aberta
        
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return Gesture.PEACE  # Paz e amor
        
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return Gesture.POINTING  # Apontando
        
        elif not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return Gesture.CLOSED_FIST  # Mão fechada
        
        return Gesture.NONE 