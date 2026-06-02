import cv2

# Abre a webcam padrão do notebook
camera = cv2.VideoCapture(0)

# Verifica se a câmera abriu corretamente
if not camera.isOpened():
    print("Erro: não foi possível acessar a webcam.")
    exit()

print("Webcam aberta com sucesso.")
print("Pressione 'q' para sair.")

while True:
    # Captura um frame da câmera
    ret, frame = camera.read()

    # Se não conseguiu capturar, encerra
    if not ret:
        print("Erro: não foi possível capturar o vídeo.")
        break

    # Espelha a imagem: fica parecendo um espelho, o que é mais intuitivo para o usuário
    frame = cv2.flip(frame, 1)

    # Mostra o vídeo em uma janela
    cv2.imshow("Webcam do notebook", frame)

    # Aguarda tecla. Se pressionar 'q', sai do loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera a câmera e fecha as janelas
camera.release()
cv2.destroyAllWindows()
