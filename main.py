import cv2
import numpy as np
from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer, Gesture
import time

def main():
    # Inicialização da câmera
    cap = cv2.VideoCapture(0)
    
    # Inicialização dos detectores
    hand_detector = HandDetector(detection_confidence=0.7)
    gesture_recognizer = GestureRecognizer()
    
    # Variáveis de controle
    previous_gesture = Gesture.NONE
    gesture_start_time = 0
    gesture_cooldown = 1.0  # Tempo mínimo entre gestos (segundos)
    is_pinching = False     # Estado do gesto de pinça
    
    # Configurações da janela
    cv2.namedWindow('Hand Gesture Control', cv2.WINDOW_NORMAL)
    
    while True:
        # Captura do frame
        success, frame = cap.read()
        if not success:
            print("Falha ao capturar imagem da câmera")
            break
            
        # Espelha a imagem horizontalmente para uma experiência mais natural
        frame = cv2.flip(frame, 1)
        
        # Detecta as mãos
        frame, results = hand_detector.find_hands(frame)
        
        # Encontra as posições dos pontos de referência
        landmarks = hand_detector.find_positions(frame)
        
        # Reconhece o gesto
        current_time = time.time()
        if landmarks:
            current_gesture = gesture_recognizer.recognize_gesture(landmarks)
            
            # Verifica se passou tempo suficiente desde o último gesto
            if (current_time - gesture_start_time) >= gesture_cooldown:
                if current_gesture != previous_gesture and current_gesture != Gesture.NONE:
                    # Executa a ação correspondente ao gesto
                    if current_gesture == Gesture.OPEN_PALM:
                        print("Ação: Pausa/Play")
                    elif current_gesture == Gesture.PEACE:
                        print("Ação: Próximo item")
                    elif current_gesture == Gesture.POINTING:
                        print("Ação: Selecionar")
                    elif current_gesture == Gesture.CLOSED_FIST:
                        print("Ação: Voltar")
                    elif current_gesture == Gesture.PINCH:
                        if not is_pinching:
                            print("Ação: Pegar objeto")
                            is_pinching = True
                        else:
                            print("Ação: Soltar objeto")
                            is_pinching = False
                    
                    gesture_start_time = current_time
                    previous_gesture = current_gesture
            
            # Exibe o gesto atual
            cv2.putText(frame, f"Gesto: {current_gesture.value}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 255, 0), 2)
            
            # Se estiver fazendo pinça, desenha um círculo entre o polegar e o indicador
            if current_gesture == Gesture.PINCH:
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                center_x = (thumb_tip[1] + index_tip[1]) // 2
                center_y = (thumb_tip[2] + index_tip[2]) // 2
                cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)
        
        # Exibe o frame
        cv2.imshow('Hand Gesture Control', frame)
        
        # Verifica se deve encerrar o programa
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpa recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma encerrado pelo usuário")
    except Exception as e:
        print(f"Erro: {str(e)}")
    finally:
        cv2.destroyAllWindows() 