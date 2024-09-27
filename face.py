import cv2
import mediapipe as mp

# Inicializa o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Função para capturar o vídeo da webcam
cap = cv2.VideoCapture(0)

# Inicializa o Face Mesh com os parâmetros
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Número máximo de rostos a serem detectados
    refine_landmarks=True,  # Inclui landmarks dos olhos e lábios refinados
    min_detection_confidence=0.5,  # Confiança mínima para detecção
    min_tracking_confidence=0.5  # Confiança mínima para o rastreamento
) as face_mesh:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Falha ao capturar a imagem")
            break
        
        # Converte a imagem de BGR para RGB (necessário para o MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Faz a detecção dos landmarks
        results = face_mesh.process(rgb_frame)

        # Se detectar algum rosto, desenha os landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

        # Mostra o frame com os landmarks
        cv2.imshow('MediaPipe Face Mesh', frame)

        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()