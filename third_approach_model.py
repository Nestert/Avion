import cv2
import numpy as np
import mediapipe as mp
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

# Определение жестов
GESTURES = ["thumbs_up", "palm", "fist", "pointing", "victory"]

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_hand_landmarks(frame):
    """Извлечение признаков руки из кадра видео с помощью MediaPipe."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    landmarks = []
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Извлечение координат всех 21 точек руки
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        return landmarks, results.multi_hand_landmarks[0]
    
    return None, None

def collect_sequence_data(video_paths, sequence_length=15):
    """Сбор последовательностей данных из видео."""
    sequences = []
    labels = []
    
    for label_idx, label in enumerate(GESTURES):
        label_dir = os.path.join(video_paths, label)
        
        if not os.path.exists(label_dir):
            print(f"Директория {label_dir} не найдена. Пропускаем.")
            continue
        
        print(f"Обработка видео для жеста: {label}")
        
        # Проход по всем видео в директории
        for video_file in os.listdir(label_dir):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
                
            video_path = os.path.join(label_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            # Сбор последовательности кадров
            frame_seq = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks, _ = extract_hand_landmarks(frame)
                
                if landmarks:
                    frame_seq.append(landmarks)
                
                # Если собрали достаточно кадров для последовательности
                if len(frame_seq) >= sequence_length:
                    sequences.append(frame_seq[-sequence_length:])
                    labels.append(label_idx)
                    
                    # Перекрытие последовательностей для увеличения данных
                    # (каждые 5 кадров начинаем новую последовательность)
                    if len(frame_seq) > sequence_length + 5:
                        frame_seq = frame_seq[-sequence_length:]
            
            cap.release()
    
    return np.array(sequences), np.array(labels)

def create_lstm_model(input_shape, num_classes):
    """Создание модели LSTM для распознавания последовательностей жестов."""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_lstm_model(sequences, labels):
    """Обучение LSTM модели на последовательностях данных."""
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.3, random_state=42
    )
    
    # Создание модели
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, features)
    model = create_lstm_model(input_shape, len(GESTURES))
    
    # Обучение модели
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Оценка модели
    start_time = time.time()
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    inference_time = (time.time() - start_time) / len(X_test)
    
    # Получение предсказаний
    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, y_pred)
    
    return model, history, accuracy, report, training_time, inference_time

def simulate_sequence_data(num_sequences=100, sequence_length=15):
    """Симуляция данных последовательностей для демонстрации."""
    print("Симуляция последовательностей данных для LSTM...")
    
    # Количество признаков (x, y, z для 21 точки руки = 63 признака)
    num_features = 63
    
    # Создание случайных последовательностей
    sequences = np.random.rand(num_sequences, sequence_length, num_features)
    
    # Создание случайных меток
    labels = np.random.randint(0, len(GESTURES), size=num_sequences)
    
    return sequences, labels

def main():
    """Основная функция для обучения LSTM модели."""
    video_dir = "gesture_videos"
    
    # Проверка наличия реальных данных
    if os.path.exists(video_dir):
        print("Сбор данных из видео...")
        sequences, labels = collect_sequence_data(video_dir)
    else:
        print(f"Директория {video_dir} не найдена. Используем симулированные данные.")
        sequences, labels = simulate_sequence_data()
    
    if len(sequences) == 0:
        print("Не удалось собрать последовательности данных.")
        return
    
    print(f"Собрано {len(sequences)} последовательностей для обучения LSTM.")
    
    # Обучение LSTM модели
    print("\nОбучение модели LSTM...")
    lstm_model, lstm_history, lstm_accuracy, lstm_report, lstm_time, lstm_inference = train_lstm_model(sequences, labels)
    
    print(f"Точность LSTM: {lstm_accuracy:.4f}")
    print("Отчет о классификации LSTM:")
    print(lstm_report)
    print(f"Время обучения LSTM: {lstm_time:.2f} секунд")
    print(f"Среднее время инференса LSTM: {lstm_inference*1000:.2f} мс")
    
    # Визуализация результатов обучения
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history.history['accuracy'], label='Обучение')
    plt.plot(lstm_history.history['val_accuracy'], label='Валидация')
    plt.title('Точность LSTM в процессе обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history.history['loss'], label='Обучение')
    plt.plot(lstm_history.history['val_loss'], label='Валидация')
    plt.title('Потери LSTM в процессе обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("lstm_training_results.png")
    plt.close()
    
    # Сохранение модели
    print("\nСохранение модели LSTM...")
    lstm_model.save("lstm_model.h5")
    
    # Сохранение результатов в CSV
    results = {
        "Метрика": ["Точность", "Время обучения (с)", "Время инференса (мс)"],
        "LSTM": [f"{lstm_accuracy:.4f}", f"{lstm_time:.2f}", f"{lstm_inference*1000:.2f}"]
    }
    
    results_df = pd.DataFrame(results)
    print("\nРезультаты LSTM модели:")
    print(results_df)
    results_df.to_csv("lstm_results.csv", index=False)
    
    print("\nОбучение LSTM модели завершено. Результаты сохранены.")

if __name__ == "__main__":
    main() 