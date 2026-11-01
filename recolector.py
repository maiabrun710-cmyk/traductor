import cv2
import mediapipe as mp
import pandas as pd
import os

# Nombre del archivo CSV donde se guardarán los datos
csv_file = "datos_dos_manos.csv"

# Configuración de MediaPipe Hands para hasta 2 manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Coloca tus manos frente a la cámara")
print("Presiona la LETRA en tu teclado correspondiente a la seña que estás haciendo")
print("Presiona 'ESC' para terminar y salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    datos_mano_1 = [0.0] * 42
    datos_mano_2 = [0.0] * 42
    manos_detectadas = False

    if results.multi_hand_landmarks:
        manos_detectadas = True
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Normalizar respecto a la muñeca (punto 0), igual que en el traductor
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            
            temp_list = []
            for lm in hand_landmarks.landmark:
                temp_list.append(lm.x - base_x)
                temp_list.append(lm.y - base_y)
            
            if idx == 0:
                datos_mano_1 = temp_list
            elif idx == 1:
                datos_mano_2 = temp_list

    # Mostrar estado en la pantalla
    estado = "Manos listas. Presiona una tecla..." if manos_detectadas else "Coloca tus manos frente a la camara"
    cv2.putText(frame, estado, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Recolector de Senas", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Tecla ESC para salir
        break
    elif manos_detectadas and key != 255:
        # Obtener la letra presionada en el teclado
        letra = chr(key).lower()
        
        # Combinar los datos de ambas manos (42 + 42 = 84 valores * 2 coordenadas = 168 características)
        fila = datos_mano_1 + datos_mano_2
        fila.append(letra) # La última columna es la clase/etiqueta
        
        # Guardar en el archivo CSV
        archivo_existe = os.path.exists(csv_file)
        df = pd.DataFrame([fila])
        df.to_csv(csv_file, mode='a', header=not archivo_existe, index=False)
        
        print(f"Muestra guardada para la letra: {letra.upper()}")

cap.release()
cv2.destroyAllWindows()
print("¡Recolección finalizada!")