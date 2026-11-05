# ============================================
# Проверка точности INT8-модели
# ============================================

import tensorflow as tf
import numpy as np

print("Загружаю тестовые данные MNIST...")
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Нормализация
x_test = x_test.astype('float32') / 255.0

print("Загружаю INT8-модель...")
interpreter = tf.lite.Interpreter(model_path="mnist_model_int8.tflite")
interpreter.allocate_tensors()

# Получаем входные и выходные тензоры
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Вход: {input_details[0]['shape']}, тип: {input_details[0]['dtype']}")
print(f"Выход: {output_details[0]['shape']}, тип: {output_details[0]['dtype']}")
print("=" * 50)

# Проверяем на первых 1000 картинках (быстрее)
correct = 0
total = 1000

print(f"Проверяю точность на {total} картинках...")

for i in range(total):
    # Берём одну картинку
    img = x_test[i]

    # Модель INT8 ждёт вход в формате int8.
    # Пиксели были 0..1, надо превратить в int8 (-128..127)
    img_int8 = (img * 255 - 128).astype(np.int8)
    img_input = np.expand_dims(img_int8, axis=(0, -1))  # (1, 28, 28, 1)

    # Подаём в модель
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    # Получаем ответ
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted = np.argmax(output)

    # Сравниваем с правильным ответом
    if predicted == y_test[i]:
        correct += 1

accuracy = correct / total * 100
print("=" * 50)
print(f"Точность INT8-модели: {accuracy:.2f}%")
print(f"Правильно: {correct} из {total}")
print("=" * 50)
print("Готово.")