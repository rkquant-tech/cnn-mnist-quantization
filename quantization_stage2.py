# ============================================
# Проект: Квантование нейросетей
# Этап 2: Создание, обучение и PTQ-квантование CNN
# ============================================

import tensorflow as tf
import numpy as np
import os

print("TensorFlow версия:", tf.__version__)
print("=" * 50)

# ---------- 1. Загрузка и подготовка MNIST ----------
print("Загружаю MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"Обучающих: {x_train.shape[0]}, тестовых: {x_test.shape[0]}")
print("=" * 50)

# ---------- 2. Создание CNN ----------
print("Создаю модель...")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
print("=" * 50)

# ---------- 3. Компиляция и обучение ----------
print("Обучаю модель (5 эпох)...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)
print("=" * 50)

# ---------- 4. Проверка точности ----------
print("Проверяю точность на тесте...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Точность FP32-модели: {test_acc * 100:.2f}%")
print("=" * 50)

# ---------- 5. Сохранение модели ----------
print("Сохраняю модель...")
model.save("mnist_model.h5")

size_fp32 = os.path.getsize("mnist_model.h5") / 1024
print(f"Размер модели FP32: {size_fp32:.1f} КБ")
print("=" * 50)

# ---------- 6. PTQ: квантование в INT8 ----------
print("Квантую модель (PTQ, INT8)...")

def representative_dataset():
    for i in range(100):
        img = x_train[i:i+1]
        yield [img.astype('float32')]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("mnist_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

size_int8 = os.path.getsize("mnist_model_int8.tflite") / 1024
print(f"Размер модели INT8: {size_int8:.1f} КБ")
print("=" * 50)

# ---------- 7. Сравнение ----------
print("РЕЗУЛЬТАТЫ:")
print(f"FP32: {size_fp32:.1f} КБ, точность {test_acc * 100:.2f}%")
print(f"INT8: {size_int8:.1f} КБ")
print(f"Сжатие: в {size_fp32 / size_int8:.1f} раз")
print("=" * 50)
print("Этап 2 завершён.")