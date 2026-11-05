# ============================================
# Этап 3: QAT (Quantization-Aware Training)
# ============================================

import os
# Принудительно заставляем TensorFlow использовать старый Keras (чтобы tfmot работал)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import numpy as np
import tf_keras as keras  # <--- Используем старый Keras для модели
import tensorflow_model_optimization as tfmot

print("TensorFlow:", tf.__version__)
print("=" * 50)

# ---------- 1. Загрузка данных ----------
print("Загружаю MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# ---------- 2. Создание модели заново ----------
print("Создаю модель с нуля через tf_keras...")
base_model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

print("Загружаю веса из старого файла mnist_model.h5...")
# Загружаем веса из старого файла. 
# by_name=False — загружаем по порядку слоев
base_model.load_weights("mnist_model.h5")

# ---------- 3. Компиляция и проверка ----------
base_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Проверяю точность загруженной модели...")
_, acc = base_model.evaluate(x_test, y_test, verbose=0)
print(f"Точность базовой модели: {acc * 100:.2f}%")
print("=" * 50)

# ---------- 4. Применяем QAT ----------
print("Оборачиваю модель в QAT...")
qat_model = tfmot.quantization.keras.quantize_model(base_model)

qat_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("=" * 50)

# ---------- 5. Дообучение с QAT ----------
print("Дообучаю модель с QAT (2 эпохи)...")
qat_model.fit(x_train, y_train, epochs=2, batch_size=128, verbose=1, validation_split=0.1)
print("=" * 50)

# ---------- 6. Проверка точности QAT-модели (float) ----------
print("Проверяю точность QAT-модели...")
_, qat_acc = qat_model.evaluate(x_test, y_test, verbose=0)
print(f"Точность QAT (float): {qat_acc * 100:.2f}%")
print("=" * 50)

# ---------- 7. Конвертация QAT-модели в INT8 ----------
print("Конвертирую QAT-модель в INT8 TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_qat_model = converter.convert()

with open("mnist_model_qat_int8.tflite", "wb") as f:
    f.write(tflite_qat_model)

size_qat = os.path.getsize("mnist_model_qat_int8.tflite") / 1024
print(f"Размер QAT INT8: {size_qat:.1f} КБ")
print("=" * 50)

# ---------- 8. Проверка точности сконвертированной INT8-модели ----------
print("Проверяю точность INT8-модели (QAT)...")
interpreter = tf.lite.Interpreter(model_path="mnist_model_qat_int8.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct = 0
total = 1000

for i in range(total):
    img = x_test[i]
    # QAT-модель ждёт FLOAT32 на входе, внутри сама квантует
    img_input = np.expand_dims(img, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    if np.argmax(output) == y_test[i]:
        correct += 1

# Считаем точность
qat_int8_acc = correct / total * 100
print(f"Точность QAT INT8: {qat_int8_acc:.2f}%")
print("=" * 50)

# ---------- 9. Итоговое сравнение ----------
print("СРАВНЕНИЕ:")
print(f"PTQ INT8: 128.3 КБ, точность 98.70% (из прошлого запуска)")
print(f"QAT INT8: {size_qat:.1f} КБ, точность {qat_int8_acc:.2f}%")
print("Этап 3 завершён.")