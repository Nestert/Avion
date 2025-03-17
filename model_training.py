import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import time

# Определение жестов
GESTURES = ["thumbs_up", "palm", "fist", "pointing", "victory"]

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_features_from_image(image_path):
    """Извлечение признаков руки из изображения с помощью MediaPipe."""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    features = []
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Извлечение координат всех 21 точек руки
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Вычисление относительных расстояний между точками
        for i in range(21):
            for j in range(i+1, 21):
                x1, y1 = hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y
                x2, y2 = hand_landmarks.landmark[j].x, hand_landmarks.landmark[j].y
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                features.append(distance)
        
        # Определение ограничивающего прямоугольника для руки
        h, w, _ = image.shape
        x_min, y_min = 1, 1
        x_max, y_max = 0, 0
        
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, landmark.x)
            y_min = min(y_min, landmark.y)
            x_max = max(x_max, landmark.x)
            y_max = max(y_max, landmark.y)
        
        # Извлечение области руки для CNN модели
        padding = 0.1
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(1, x_max + padding)
        y_max = min(1, y_max + padding)
        
        x_min, y_min = int(x_min * w), int(y_min * h)
        x_max, y_max = int(x_max * w), int(y_max * h)
        
        # Проверка валидности координат
        if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0 or x_max > w or y_max > h:
            # Случай некорректных координат, используем всё изображение
            hand_crop = cv2.resize(image, (128, 128))
        else:
            hand_crop = image[y_min:y_max, x_min:x_max]
            hand_crop = cv2.resize(hand_crop, (128, 128))
        
        return features, hand_crop
    
    return None, None

def collect_dataset(data_dir):
    """Сбор данных из каталогов с изображениями."""
    features_data = []
    labels = []
    
    image_data = []
    image_labels = []
    
    # Проход по всем жестам
    for gesture in GESTURES:
        gesture_dir = os.path.join(data_dir, gesture)
        
        # Проверка наличия директории
        if not os.path.exists(gesture_dir):
            print(f"Директория {gesture_dir} не найдена. Пропускаем.")
            continue
        
        print(f"Обработка жеста: {gesture}")
        
        # Проход по всем изображениям в директории
        for image_file in os.listdir(gesture_dir):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            image_path = os.path.join(gesture_dir, image_file)
            features, hand_crop = extract_features_from_image(image_path)
            
            if features is not None and hand_crop is not None:
                features_data.append(features)
                labels.append(gesture)
                
                # Нормализация данных для CNN
                hand_crop = hand_crop / 255.0
                image_data.append(hand_crop)
                image_labels.append(gesture)
    
    # Кодирование меток
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    encoded_image_labels = le.transform(image_labels)
    
    return features_data, np.array(encoded_labels), np.array(image_data), np.array(encoded_image_labels), le

def train_random_forest(features, labels):
    """Обучение модели случайного леса."""
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Создание и обучение модели
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Оценка модели
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return model, accuracy, report, training_time, inference_time

def train_cnn(images, labels):
    """Обучение сверточной нейронной сети."""
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
    
    # Преобразование меток в one-hot encoding
    y_train_one_hot = tf.keras.utils.to_categorical(y_train)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test)
    
    # Создание модели
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(labels)), activation='softmax')
    ])
    
    # Компиляция модели
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Обучение модели
    start_time = time.time()
    history = model.fit(
        X_train, y_train_one_hot,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test_one_hot),
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Оценка модели
    start_time = time.time()
    _, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
    inference_time = (time.time() - start_time) / len(X_test)
    
    # Получение предсказаний в формате классов, а не one-hot encoding
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred)
    
    return model, history, accuracy, report, training_time, inference_time

def visualize_results(rf_accuracy, cnn_history, rf_time, cnn_time, rf_inference, cnn_inference):
    """Визуализация результатов обучения моделей."""
    # Создание фигуры с подграфиками
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # График точности CNN во время обучения
    axes[0, 0].plot(cnn_history.history['accuracy'], label='Обучение')
    axes[0, 0].plot(cnn_history.history['val_accuracy'], label='Валидация')
    axes[0, 0].set_title('Точность CNN в процессе обучения')
    axes[0, 0].set_xlabel('Эпоха')
    axes[0, 0].set_ylabel('Точность')
    axes[0, 0].legend()
    
    # График потерь CNN во время обучения
    axes[0, 1].plot(cnn_history.history['loss'], label='Обучение')
    axes[0, 1].plot(cnn_history.history['val_loss'], label='Валидация')
    axes[0, 1].set_title('Потери CNN в процессе обучения')
    axes[0, 1].set_xlabel('Эпоха')
    axes[0, 1].set_ylabel('Потери')
    axes[0, 1].legend()
    
    # Сравнение точности моделей
    accuracies = [rf_accuracy, cnn_history.history['val_accuracy'][-1]]
    axes[1, 0].bar(['Random Forest', 'CNN'], accuracies)
    axes[1, 0].set_title('Сравнение точности моделей')
    axes[1, 0].set_ylabel('Точность')
    for i, v in enumerate(accuracies):
        axes[1, 0].text(i, v + 0.01, f"{v:.2f}", ha='center')
    
    # Сравнение времени обучения и инференса
    train_times = [rf_time, cnn_time]
    inference_times = [rf_inference * 1000, cnn_inference * 1000]  # в миллисекундах
    
    ax1 = axes[1, 1]
    ax1.bar(['Random Forest', 'CNN'], train_times, color='blue', alpha=0.7, label='Время обучения (с)')
    ax1.set_title('Сравнение временных характеристик')
    ax1.set_ylabel('Время обучения (с)')
    for i, v in enumerate(train_times):
        ax1.text(i, v + 0.5, f"{v:.2f}с", ha='center')
    
    ax2 = ax1.twinx()
    ax2.bar(['Random Forest', 'CNN'], inference_times, color='red', alpha=0.3, label='Время инференса (мс)')
    ax2.set_ylabel('Время инференса (мс)')
    
    # Объединение легенд
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig("results_comparison.png")
    plt.close()

def main():
    """Основная функция для обучения моделей и сохранения результатов."""
    data_dir = "gesture_data"
    
    # Проверка наличия папки с данными
    if not os.path.exists(data_dir):
        print(f"Директория {data_dir} не найдена. Создаём её...")
        os.makedirs(data_dir)
        
        # Создаём подпапки для каждого жеста
        for gesture in GESTURES:
            os.makedirs(os.path.join(data_dir, gesture), exist_ok=True)
        
        print(f"Создали директории для сбора данных в {data_dir}.")
        print("Пожалуйста, поместите изображения жестов в соответствующие папки и запустите скрипт снова.")
        return
    
    print("Сбор данных из изображений...")
    features_data, encoded_labels, image_data, encoded_image_labels, label_encoder = collect_dataset(data_dir)
    
    if len(features_data) == 0 or len(image_data) == 0:
        print("Не удалось собрать данные. Проверьте наличие изображений в папках.")
        return
    
    print(f"Собрано {len(features_data)} образцов для обучения Random Forest.")
    print(f"Собрано {len(image_data)} образцов для обучения CNN.")
    
    # Обучение Random Forest
    print("\nОбучение модели Random Forest...")
    rf_model, rf_accuracy, rf_report, rf_time, rf_inference = train_random_forest(features_data, encoded_labels)
    print(f"Точность Random Forest: {rf_accuracy:.4f}")
    print("Отчет о классификации Random Forest:")
    print(rf_report)
    print(f"Время обучения Random Forest: {rf_time:.2f} секунд")
    print(f"Среднее время инференса Random Forest: {rf_inference*1000:.2f} мс")
    
    # Обучение CNN
    print("\nОбучение модели CNN...")
    cnn_model, cnn_history, cnn_accuracy, cnn_report, cnn_time, cnn_inference = train_cnn(image_data, encoded_image_labels)
    print(f"Точность CNN: {cnn_accuracy:.4f}")
    print("Отчет о классификации CNN:")
    print(cnn_report)
    print(f"Время обучения CNN: {cnn_time:.2f} секунд")
    print(f"Среднее время инференса CNN: {cnn_inference*1000:.2f} мс")
    
    # Визуализация результатов
    print("\nВизуализация результатов...")
    visualize_results(rf_accuracy, cnn_history, rf_time, cnn_time, rf_inference, cnn_inference)
    
    # Сохранение моделей
    print("\nСохранение моделей...")
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump((rf_model, label_encoder), f)
    
    cnn_model.save("cnn_model.h5")
    
    # Сохранение информации о классах
    with open("class_info.pkl", "wb") as f:
        pickle.dump({"classes": label_encoder.classes_.tolist()}, f)
    
    # Сводная таблица результатов
    results = {
        "Метрика": ["Точность", "Время обучения (с)", "Время инференса (мс)"],
        "Random Forest": [f"{rf_accuracy:.4f}", f"{rf_time:.2f}", f"{rf_inference*1000:.2f}"],
        "CNN": [f"{cnn_accuracy:.4f}", f"{cnn_time:.2f}", f"{cnn_inference*1000:.2f}"]
    }
    
    results_df = pd.DataFrame(results)
    print("\nСравнение подходов:")
    print(results_df)
    results_df.to_csv("results_comparison.csv", index=False)
    
    print("\nОбучение моделей завершено. Результаты сохранены.")

if __name__ == "__main__":
    main() 