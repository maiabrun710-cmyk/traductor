import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

# 1. Cargar los datos
csv_file = "datos_dos_manos.csv"
if not pd.io.common.file_exists(csv_file):
    print(f"No se encontró el archivo {csv_file}. ¡Primero debes recolectar muestras!")
    exit()

df = pd.read_csv(csv_file)

# Separar características (X) y etiquetas (y)
X = df.iloc[:, :-1].values  # Todas las columnas menos la última
y = df.iloc[:, -1].values   # La última columna ('clase')

# 2. Codificar las letras (a, b, c...) a números (0, 1, 2...)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)

# Guardar el codificador para usarlo en la predicción
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# 3. Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. Crear la Red Neuronal
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Entrenar el modelo
print("Entrenando el modelo...")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# 6. Guardar el modelo entrenado
model.save("modelo_senas.h5")
print("¡Modelo entrenado y guardado con éxito como 'modelo_senas.h5'!")