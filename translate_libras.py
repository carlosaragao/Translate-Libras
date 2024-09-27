import cv2
import mediapipe as mp

# Inicializar o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_hands_and_translate():
    # Capturar vídeo da webcam
    cap = cv2.VideoCapture(0)  # 0 significa que está usando a webcam padrão

    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return

    # Função para mapear landmarks para letras ou palavras
    def translate_gesture(landmarks):
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP].x
        
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
        index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y
        ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
        
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
        pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP].y

        # Verificando a posição dos dedos para a letra "A"
        if (thumb_tip < thumb_mcp and  # Polegar ao lado da mão
            index_tip > index_pip and  # Dedo indicador dobrado
            middle_tip > middle_pip and  # Dedo médio dobrado
            ring_tip > ring_pip and  # Dedo anelar dobrado
            pinky_tip > pinky_pip):  # Dedo mínimo dobrado
            return 'A'
        
        # Verificando a posição dos dedos para a letra "B"
        if (thumb_tip < thumb_mcp and  # Polegar dobrado
            index_tip < index_pip and  # Dedo indicador esticado
            middle_tip < middle_pip and  # Dedo médio esticado
            ring_tip < ring_pip and  # Dedo anelar esticado
            pinky_tip < pinky_pip):  # Dedo mínimo esticado
            return 'B'

        # Adicionar outras letras conforme necessário
        return 'Desconhecido'

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao capturar o frame.")
            break

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processar o frame para detectar as mãos
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar landmarks das mãos
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Traduzir gestos com base nos landmarks
                gesture = translate_gesture(hand_landmarks.landmark)
                
                # Exibir a tradução do gesto no frame
                cv2.putText(frame, f'Gesto: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Exibir o frame processado
        cv2.imshow('Webcam', frame)

        # Pressionar 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a webcam e fechar as janelas
    cap.release()
    cv2.destroyAllWindows()

# Iniciar a detecção de mãos e tradução via webcam
detect_hands_and_translate()