import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "face_landmarker.task"


def draw_landmarks_on_frame(frame, detection_result):
    """
    Desenha os landmarks faciais encontrados pelo MediaPipe.
    """
    height, width, _ = frame.shape

    if not detection_result.face_landmarks:
        return frame

    for face_landmarks in detection_result.face_landmarks:
        for landmark in face_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    return frame


# Configura o modelo do Face Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

detector = vision.FaceLandmarker.create_from_options(options)

# Abre a webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: não foi possível acessar a webcam.")
    exit()

print("Webcam aberta com sucesso.")
print("Pressione 'q' para sair.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Erro: não foi possível capturar o vídeo.")
        break

    # Espelha a imagem: fica parecendo um espelho, o que é mais intuitivo para o usuário
    frame = cv2.flip(frame, 1)

    # OpenCV usa BGR; MediaPipe usa RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Cria imagem MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Detecta landmarks
    detection_result = detector.detect(mp_image)

    # Desenha landmarks
    frame = draw_landmarks_on_frame(frame, detection_result)

    cv2.imshow("MediaPipe Face Landmarker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
detector.close()
