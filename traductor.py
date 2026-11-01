import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Cargar el modelo entrenado y el codificador de etiquetas
model = tf.keras.models.load_model("modelo_senas.h5", compile=False)
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Configuración de MediaPipe Hands para hasta 2 manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("--- TRADUCTOR EN TIEMPO REAL ---")
print("Coloca tus manos frente a la cámara. Presiona 'ESC' para salir.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    datos_mano_1 = [0.0] * 42
    datos_mano_2 = [0.0] * 42
    letra_predicha = "Esperando..."
    confianza = 0.0

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Normalizar respecto a la muñeca (punto 0)
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

        # Unir los datos de ambas manos y darles forma para el modelo (1 fila, 168 columnas)
        input_data = np.array([datos_mano_1 + datos_mano_2])

        # Hacer la predicción
        prediccion = model.predict(input_data, verbose=0)
        clase_idx = np.argmax(prediccion[0])
        confianza = float(np.max(prediccion[0]))

        # Si la confianza es alta (> 75%), mostrar la letra
        if confianza > 0.75:
            letra_predicha = encoder.inverse_transform([clase_idx])[0].upper()
        else:
            letra_predicha = "?"

    # Mostrar la seña traducida en la pantalla
    texto_mostrar = f"Letra: {letra_predicha} ({confianza*100:.1f}%)"
    cv2.putText(frame, texto_mostrar, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Traductor de Senas - Dos Manos", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()