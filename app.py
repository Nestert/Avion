import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
import tempfile
import os
import time
from datetime import datetime
import collections
import pickle
import torch
import torch.nn as nn

# Настройки страницы
st.set_page_config(
    page_title="Gesture Recognition",
    page_icon="👋",
    layout="wide"
)

# Заголовок приложения
st.title("Gesture Recognition")
st.write("This system recognizes 5 gestures: thumbs up, palm, fist, pointing finger, victory sign (V)")

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Список доступных жестов
GESTURE_CLASSES = ["Thumbs Up", "Palm", "Fist", "Pointing", "Victory"]

# Классы моделей для PyTorch
class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        
        # Сверточные слои
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Полносвязные слои
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Преобразование в плоский вектор
        x = self.fc_layers(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=1, num_classes=5, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM слои
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, 
                            batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Второй LSTM имеет входной размер, равный удвоенному размеру скрытого состояния из-за двунаправленности
        self.lstm2 = nn.LSTM(hidden_size*2, hidden_size, num_layers=1, 
                            batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Нормализация для стабилизации обучения
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Полносвязные слои для классификации
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Первый LSTM слой (двунаправленный)
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Второй LSTM слой
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # Берем только выход последнего временного шага
        out = out[:, -1, :]
        out = self.layer_norm(out)
        
        # Полносвязные слои
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out

# Загрузка моделей
@st.cache_resource
def load_models():
    models = {}
    
    # Загрузка Random Forest
    try:
        with open('models/random_forest_model.pkl', 'rb') as f:
            rf_model, label_encoder = pickle.load(f)
        models["rf"] = (rf_model, label_encoder)
        print("Random Forest модель загружена")
    except Exception as e:
        print(f"Ошибка загрузки Random Forest: {e}")
        models["rf"] = None
    
    # Загрузка CNN
    try:
        # Загрузка информации о классах
        with open('models/class_info.pkl', 'rb') as f:
            class_info = pickle.load(f)
        
        # Создание и загрузка модели CNN
        cnn_model = CNNModel(num_classes=len(class_info["classes"]))
        cnn_model.load_state_dict(torch.load('models/cnn_model.pth', map_location=torch.device('cpu')))
        cnn_model.eval()  # Перевод в режим оценки
        models["cnn"] = (cnn_model, class_info["classes"])
        print("CNN модель загружена")
    except Exception as e:
        print(f"Ошибка загрузки CNN: {e}")
        models["cnn"] = None
    
    # Загрузка LSTM
    try:
        # Создание и загрузка модели LSTM
        lstm_model = LSTMModel(input_size=63, hidden_size=128, num_layers=1, num_classes=len(class_info["classes"]), dropout_rate=0.3)
        lstm_model.load_state_dict(torch.load('models/lstm_model.pth', map_location=torch.device('cpu')))
        lstm_model.eval()  # Перевод в режим оценки
        models["lstm"] = (lstm_model, class_info["classes"])
        print("LSTM модель загружена")
    except Exception as e:
        print(f"Ошибка загрузки LSTM: {e}")
        models["lstm"] = None
    
    return models

# Загружаем модели при запуске приложения
MODELS = load_models()

# Класс для стабилизации предсказаний через систему голосования
class PredictionStabilizer:
    def __init__(self, window_size=10, threshold=0.6):
        self.window_size = window_size
        self.threshold = threshold
        self.predictions = collections.deque(maxlen=window_size)
        self.current_prediction = "Not detected"
        self.current_confidence = 0.0
        
    def update(self, new_prediction, new_confidence):
        # Если новое предсказание - "Not detected" или "Error", не обновляем историю
        if new_prediction in ["Not detected", "Error"]:
            if len(self.predictions) == 0:
                self.current_prediction = new_prediction
                self.current_confidence = new_confidence
            return self.current_prediction, self.current_confidence
        
        # Добавляем новое предсказание в историю
        self.predictions.append(new_prediction)
        
        # Подсчитываем частоту каждого предсказания
        prediction_counts = collections.Counter(self.predictions)
        
        # Находим наиболее частое предсказание
        most_common = prediction_counts.most_common(1)
        
        # Проверяем, достаточно ли уверенности для смены предсказания
        if most_common:
            prediction, count = most_common[0]
            confidence_ratio = count / len(self.predictions)
            
            # Обновляем текущее предсказание, если частота превышает порог
            if confidence_ratio >= self.threshold:
                self.current_prediction = prediction
                self.current_confidence = confidence_ratio
        
        return self.current_prediction, self.current_confidence
        
    def clear(self):
        self.predictions.clear()
        self.current_prediction = "Not detected"
        self.current_confidence = 0.0

# Функция для выделения признаков руки с помощью MediaPipe
def extract_hand_features(image):
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
        
        return features, hand_landmarks
    
    return None, None

# Класс для хранения истории признаков для LSTM
class LandmarkHistory:
    def __init__(self, max_length=15):
        self.max_length = max_length
        self.landmarks_history = collections.deque(maxlen=max_length)
        
    def add(self, landmarks):
        self.landmarks_history.append(landmarks)
        
    def get_sequence(self):
        if len(self.landmarks_history) < self.max_length:
            return None
        return np.array(list(self.landmarks_history))
        
    def clear(self):
        self.landmarks_history.clear()

# Функции для разных подходов к распознаванию жестов
def approach_1_random_forest(features):
    """Подход 1: Случайный лес на основе координат и расстояний"""
    try:
        if MODELS["rf"] is not None:
            # Используем реальную модель
            rf_model, label_encoder = MODELS["rf"]
            prediction_idx = rf_model.predict([features])[0]
            prediction = label_encoder.inverse_transform([prediction_idx])[0]
            
            # Получаем вероятности
            probabilities = rf_model.predict_proba([features])[0]
            confidence = probabilities[prediction_idx]
            
            # Преобразуем имя жеста к формату интерфейса
            prediction = prediction.replace('_', ' ').title()
            
            return prediction, confidence
        else:
            # Симуляция предсказания для демонстрации
            seed = int(sum(features[:5]) * 1000) % 100000
            np.random.seed(seed)
            
            gesture_list = ["Thumbs Up", "Palm", "Fist", "Pointing", "Victory"]
            prediction = gesture_list[np.random.randint(0, 5)]
            confidence = np.random.uniform(0.7, 0.99)
            
            return prediction, confidence
    except Exception as e:
        st.error(f"Error in approach 1: {e}")
        return "Error", 0

def approach_2_cnn(image, hand_landmarks):
    """Подход 2: Сверточная нейронная сеть с обработкой изображения"""
    try:
        if MODELS["cnn"] is not None and hand_landmarks:
            # Используем реальную модель
            cnn_model, classes = MODELS["cnn"]
            
            # Подготовка изображения для CNN
            h, w, _ = image.shape
            
            # Определение ограничивающего прямоугольника для руки
            x_min, y_min = 1, 1
            x_max, y_max = 0, 0
            
            for landmark in hand_landmarks.landmark:
                x_min = min(x_min, landmark.x)
                y_min = min(y_min, landmark.y)
                x_max = max(x_max, landmark.x)
                y_max = max(y_max, landmark.y)
            
            # Добавляем отступ
            padding = 0.1
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(1, x_max + padding)
            y_max = min(1, y_max + padding)
            
            x_min, y_min = int(x_min * w), int(y_min * h)
            x_max, y_max = int(x_max * w), int(y_max * h)
            
            # Обрезка и изменение размера изображения
            if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                hand_crop = cv2.resize(image, (128, 128))
            else:
                hand_crop = image[y_min:y_max, x_min:x_max]
                hand_crop = cv2.resize(hand_crop, (128, 128))
            
            # Нормализация
            hand_crop = hand_crop / 255.0
            
            # Преобразование в тензор и изменение формата (batch, channels, height, width)
            hand_crop = np.transpose(hand_crop, (2, 0, 1))
            hand_crop = np.expand_dims(hand_crop, axis=0)
            hand_crop_tensor = torch.FloatTensor(hand_crop)
            
            # Предсказание
            with torch.no_grad():
                outputs = cnn_model(hand_crop_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                prediction_idx = torch.argmax(probabilities).item()
                confidence = probabilities[prediction_idx].item()
            
            # Преобразуем имя жеста к формату интерфейса
            prediction = classes[prediction_idx].replace('_', ' ').title()
            
            return prediction, confidence
        else:
            # Симуляция предсказания для демонстрации
            if hand_landmarks:
                coords = [hand_landmarks.landmark[i].x + hand_landmarks.landmark[i].y for i in range(5)]
                seed = int(sum(coords) * 1000) % 100000
                np.random.seed(seed)
            
            gesture_list = ["Thumbs Up", "Palm", "Fist", "Pointing", "Victory"]
            prediction = gesture_list[np.random.randint(0, 5)]
            confidence = np.random.uniform(0.6, 0.95)
            
            return prediction, confidence
    except Exception as e:
        st.error(f"Error in approach 2: {e}")
        return "Error", 0

def approach_3_lstm(landmark_sequence):
    """Подход 3: LSTM на временных последовательностях"""
    try:
        if MODELS["lstm"] is not None:
            # Используем реальную модель
            lstm_model, classes = MODELS["lstm"]
            
            # Загрузка скейлера (если есть)
            scaler = None
            try:
                with open('models/lstm_scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as e:
                print(f"Скейлер не найден, используем ненормализованные данные: {e}")
            
            # Нормализация последовательности данных, если скейлер доступен
            if scaler is not None:
                # Получаем размерность последовательности
                seq_length, num_features = landmark_sequence.shape
                
                # Преобразуем в 2D для нормализации
                sequence_reshaped = landmark_sequence.reshape(seq_length * num_features)
                sequence_normalized = scaler.transform(sequence_reshaped.reshape(-1, num_features))
                # Возвращаем к исходной форме
                landmark_sequence = sequence_normalized
            
            # Преобразование последовательности в тензор
            sequence_tensor = torch.FloatTensor(landmark_sequence).unsqueeze(0)  # добавляем размерность батча
            
            # Предсказание
            with torch.no_grad():
                outputs = lstm_model(sequence_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                prediction_idx = torch.argmax(probabilities).item()
                confidence = probabilities[prediction_idx].item()
            
            # Преобразуем имя жеста к формату интерфейса
            prediction = classes[prediction_idx].replace('_', ' ').title()
            
            return prediction, confidence
        else:
            # Симуляция предсказания для демонстрации
            last_features = landmark_sequence[-1]
            seed = int(sum(last_features[:5]) * 1000) % 100000
            np.random.seed(seed)
            
            gesture_list = ["Thumbs Up", "Palm", "Fist", "Pointing", "Victory"]
            prediction = gesture_list[np.random.randint(0, 5)]
            confidence = np.random.uniform(0.65, 0.97)
            
            return prediction, confidence
    except Exception as e:
        st.error(f"Error in approach 3: {e}")
        return "Error", 0

# Функция для отображения результатов распознавания
def display_results(image, hand_landmarks, prediction1, confidence1, prediction2, confidence2, prediction3, confidence3, evaluation_mode=False, current_gesture=None):
    h, w, c = image.shape
    
    # Отрисовка меток руки, если они обнаружены
    if hand_landmarks:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            image_rgb, 
            hand_landmarks, 
            mp_hands.HAND_CONNECTIONS
        )
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Добавление текста с предсказаниями
    cv2.putText(image, f"RF: {prediction1} ({confidence1:.2f})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, f"CNN: {prediction2} ({confidence2:.2f})", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"LSTM: {prediction3} ({confidence3:.2f})", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Отображение текущего режима и выбранного жеста в режиме оценки
    if evaluation_mode and current_gesture:
        cv2.putText(image, f"Evaluation Mode: {current_gesture}", (10, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image

# Основная функция для обработки видеопотока
def process_video(video_source):
    # Метрики для подходов
    metrics = {
        "RF": {"correct": 0, "total": 0, "time": 0},
        "CNN": {"correct": 0, "total": 0, "time": 0},
        "LSTM": {"correct": 0, "total": 0, "time": 0}
    }
    
    # Инициализация хранилища для LSTM
    landmark_history = LandmarkHistory(max_length=15)
    
    # Инициализация стабилизаторов предсказаний
    rf_stabilizer = PredictionStabilizer(window_size=stability_window, threshold=stability_threshold)
    cnn_stabilizer = PredictionStabilizer(window_size=stability_window, threshold=stability_threshold)
    lstm_stabilizer = PredictionStabilizer(window_size=stability_window, threshold=stability_threshold)
    
    # Для веб-камеры
    if video_source == 0:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        # Колонки для отображения метрик
        col1, col2, col3 = st.columns(3)
        rf_metrics = col1.empty()
        cnn_metrics = col2.empty()
        lstm_metrics = col3.empty()
        
        # Создаем контейнер для кнопки остановки один раз за пределами цикла
        stop_button_container = st.empty()
        stop_pressed = stop_button_container.button("Stop", key="webcam_stop_button")
        
        # Флаг для отслеживания состояния остановки
        running = True
        
        while cap.isOpened() and running and not stop_pressed:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Обработка кадра
            features, hand_landmarks = extract_hand_features(frame)
            
            prediction1, confidence1 = "Not detected", 0
            prediction2, confidence2 = "Not detected", 0
            prediction3, confidence3 = "Not detected", 0
            
            if features:
                # Добавление признаков в историю для LSTM
                landmark_history.add(features[:63])  # Только координаты ключевых точек
                
                # Замер времени для Random Forest
                start_time = time.time()
                raw_prediction1, raw_confidence1 = approach_1_random_forest(features)
                rf_time = time.time() - start_time
                metrics["RF"]["time"] += rf_time
                metrics["RF"]["total"] += 1
                
                # Стабилизация предсказания Random Forest
                prediction1, confidence1 = rf_stabilizer.update(raw_prediction1, raw_confidence1)
                
                # Замер времени для CNN
                start_time = time.time()
                raw_prediction2, raw_confidence2 = approach_2_cnn(frame, hand_landmarks)
                cnn_time = time.time() - start_time
                metrics["CNN"]["time"] += cnn_time
                metrics["CNN"]["total"] += 1
                
                # Стабилизация предсказания CNN
                prediction2, confidence2 = cnn_stabilizer.update(raw_prediction2, raw_confidence2)
                
                # Замер времени для LSTM, если достаточно кадров в истории
                sequence = landmark_history.get_sequence()
                if sequence is not None:
                    start_time = time.time()
                    raw_prediction3, raw_confidence3 = approach_3_lstm(sequence)
                    lstm_time = time.time() - start_time
                    metrics["LSTM"]["time"] += lstm_time
                    metrics["LSTM"]["total"] += 1
                    
                    # Стабилизация предсказания LSTM
                    prediction3, confidence3 = lstm_stabilizer.update(raw_prediction3, raw_confidence3)
                    
                    # Обновление метрик для LSTM в режиме оценки
                    if evaluation_mode and current_gesture:
                        # Проверяем, совпадает ли предсказание с выбранным жестом
                        metrics["LSTM"]["total"] += 1
                        if prediction3 == current_gesture:
                            metrics["LSTM"]["correct"] += 1
                
                    # Обновление метрик для RF и CNN в режиме оценки
                    if evaluation_mode and current_gesture:
                        # Random Forest
                        metrics["RF"]["total"] += 1
                        if prediction1 == current_gesture:
                            metrics["RF"]["correct"] += 1
                        
                        # CNN
                        metrics["CNN"]["total"] += 1
                        if prediction2 == current_gesture:
                            metrics["CNN"]["correct"] += 1
                    
                    # Если не в режиме оценки, используем консенсус между моделями для оценки точности
                    elif not evaluation_mode:
                        # Определяем консенсус (большинство голосов)
                        predictions = [prediction1, prediction2]
                        if prediction3 != "Not detected" and prediction3 != "Error":
                            predictions.append(prediction3)
                        
                        if len(predictions) >= 2:  # Если есть хотя бы два предсказания
                            prediction_counts = collections.Counter(predictions)
                            most_common = prediction_counts.most_common(1)
                            
                            if most_common:
                                consensus, count = most_common[0]
                                
                                # Если большинство моделей согласны
                                if count >= len(predictions)/2:
                                    # Random Forest
                                    metrics["RF"]["total"] += 1
                                    if prediction1 == consensus:
                                        metrics["RF"]["correct"] += 1
                                    
                                    # CNN
                                    metrics["CNN"]["total"] += 1
                                    if prediction2 == consensus:
                                        metrics["CNN"]["correct"] += 1
                                    
                                    # LSTM (если активен)
                                    if prediction3 != "Not detected" and prediction3 != "Error":
                                        metrics["LSTM"]["total"] += 1
                                        if prediction3 == consensus:
                                            metrics["LSTM"]["correct"] += 1
            else:
                # Если рука не обнаружена, очищаем историю и стабилизаторы
                landmark_history.clear()
                rf_stabilizer.clear()
                cnn_stabilizer.clear()
                lstm_stabilizer.clear()
            
            # Отображение результатов на кадре
            result_frame = display_results(frame, hand_landmarks, prediction1, confidence1, 
                                          prediction2, confidence2, prediction3, confidence3, evaluation_mode, current_gesture)
            
            # Преобразование для отображения в Streamlit
            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            stframe.image(result_frame_rgb, channels="RGB")
            
            # Обновление метрик
            if metrics["RF"]["total"] > 0:
                rf_accuracy = metrics["RF"]["correct"] / metrics["RF"]["total"]
                rf_avg_time = metrics["RF"]["time"] / metrics["RF"]["total"]
                rf_metrics.markdown(f"""
                ### Random Forest Metrics:
                - Accuracy: {rf_accuracy:.2f}
                - Average time: {rf_avg_time*1000:.2f} ms
                """)
            
            if metrics["CNN"]["total"] > 0:
                cnn_accuracy = metrics["CNN"]["correct"] / metrics["CNN"]["total"]
                cnn_avg_time = metrics["CNN"]["time"] / metrics["CNN"]["total"]
                cnn_metrics.markdown(f"""
                ### CNN Metrics:
                - Accuracy: {cnn_accuracy:.2f}
                - Average time: {cnn_avg_time*1000:.2f} ms
                """)
                
            if metrics["LSTM"]["total"] > 0:
                lstm_accuracy = metrics["LSTM"]["correct"] / metrics["LSTM"]["total"]
                lstm_avg_time = metrics["LSTM"]["time"] / metrics["LSTM"]["total"]
                lstm_metrics.markdown(f"""
                ### LSTM Metrics:
                - Accuracy: {lstm_accuracy:.2f}
                - Average time: {lstm_avg_time*1000:.2f} ms
                """)
                
            # Проверяем, была ли нажата кнопка остановки
            # Мы не создаем новую кнопку в цикле, а просто проверяем состояние
            if stop_pressed:
                running = False
            
            # Добавляем небольшую задержку для уменьшения нагрузки на CPU
            time.sleep(0.01)
            
        cap.release()
        
    # Для загруженного видео
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_source.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        # Информация о видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Прогресс бар
        progress_bar = st.progress(0)
        stframe = st.empty()
        
        # Колонки для отображения метрик
        col1, col2, col3 = st.columns(3)
        rf_metrics = col1.empty()
        cnn_metrics = col2.empty()
        lstm_metrics = col3.empty()
        
        # Создаем контейнер для кнопки остановки один раз за пределами цикла
        stop_button_container = st.empty()
        stop_pressed = stop_button_container.button("Stop", key="video_stop_button")
        
        frame_count = 0
        
        while cap.isOpened() and not stop_pressed:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Обработка каждого 3-го кадра для ускорения
            if frame_count % 3 == 0:
                # Обработка кадра
                features, hand_landmarks = extract_hand_features(frame)
                
                prediction1, confidence1 = "Not detected", 0
                prediction2, confidence2 = "Not detected", 0
                prediction3, confidence3 = "Not detected", 0
                
                if features:
                    # Добавление признаков в историю для LSTM
                    landmark_history.add(features[:63])  # Только координаты ключевых точек
                    
                    # Замер времени для Random Forest
                    start_time = time.time()
                    raw_prediction1, raw_confidence1 = approach_1_random_forest(features)
                    rf_time = time.time() - start_time
                    metrics["RF"]["time"] += rf_time
                    metrics["RF"]["total"] += 1
                    
                    # Стабилизация предсказания Random Forest
                    prediction1, confidence1 = rf_stabilizer.update(raw_prediction1, raw_confidence1)
                    
                    # Замер времени для CNN
                    start_time = time.time()
                    raw_prediction2, raw_confidence2 = approach_2_cnn(frame, hand_landmarks)
                    cnn_time = time.time() - start_time
                    metrics["CNN"]["time"] += cnn_time
                    metrics["CNN"]["total"] += 1
                    
                    # Стабилизация предсказания CNN
                    prediction2, confidence2 = cnn_stabilizer.update(raw_prediction2, raw_confidence2)
                    
                    # Замер времени для LSTM, если достаточно кадров в истории
                    sequence = landmark_history.get_sequence()
                    if sequence is not None:
                        start_time = time.time()
                        raw_prediction3, raw_confidence3 = approach_3_lstm(sequence)
                        lstm_time = time.time() - start_time
                        metrics["LSTM"]["time"] += lstm_time
                        metrics["LSTM"]["total"] += 1
                        
                        # Стабилизация предсказания LSTM
                        prediction3, confidence3 = lstm_stabilizer.update(raw_prediction3, raw_confidence3)
                        
                        # Обновление метрик для LSTM в режиме оценки
                        if evaluation_mode and current_gesture:
                            # Проверяем, совпадает ли предсказание с выбранным жестом
                            metrics["LSTM"]["total"] += 1
                            if prediction3 == current_gesture:
                                metrics["LSTM"]["correct"] += 1
                    
                    # Обновление метрик для RF и CNN в режиме оценки
                    if evaluation_mode and current_gesture:
                        # Random Forest
                        metrics["RF"]["total"] += 1
                        if prediction1 == current_gesture:
                            metrics["RF"]["correct"] += 1
                        
                        # CNN
                        metrics["CNN"]["total"] += 1
                        if prediction2 == current_gesture:
                            metrics["CNN"]["correct"] += 1
                    
                    # Если не в режиме оценки, используем консенсус между моделями для оценки точности
                    elif not evaluation_mode:
                        # Определяем консенсус (большинство голосов)
                        predictions = [prediction1, prediction2]
                        if prediction3 != "Not detected" and prediction3 != "Error":
                            predictions.append(prediction3)
                        
                        if len(predictions) >= 2:  # Если есть хотя бы два предсказания
                            prediction_counts = collections.Counter(predictions)
                            most_common = prediction_counts.most_common(1)
                            
                            if most_common:
                                consensus, count = most_common[0]
                                
                                # Если большинство моделей согласны
                                if count >= len(predictions)/2:
                                    # Random Forest
                                    metrics["RF"]["total"] += 1
                                    if prediction1 == consensus:
                                        metrics["RF"]["correct"] += 1
                                    
                                    # CNN
                                    metrics["CNN"]["total"] += 1
                                    if prediction2 == consensus:
                                        metrics["CNN"]["correct"] += 1
                                    
                                    # LSTM (если активен)
                                    if prediction3 != "Not detected" and prediction3 != "Error":
                                        metrics["LSTM"]["total"] += 1
                                        if prediction3 == consensus:
                                            metrics["LSTM"]["correct"] += 1
                else:
                    # Если рука не обнаружена, очищаем историю и стабилизаторы
                    landmark_history.clear()
                    rf_stabilizer.clear()
                    cnn_stabilizer.clear()
                    lstm_stabilizer.clear()
                
                # Отображение результатов на кадре
                result_frame = display_results(frame, hand_landmarks, prediction1, confidence1, 
                                              prediction2, confidence2, prediction3, confidence3, evaluation_mode, current_gesture)
                
                # Преобразование для отображения в Streamlit
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                stframe.image(result_frame_rgb, channels="RGB")
                
                # Обновление метрик
                if metrics["RF"]["total"] > 0:
                    rf_accuracy = metrics["RF"]["correct"] / metrics["RF"]["total"]
                    rf_avg_time = metrics["RF"]["time"] / metrics["RF"]["total"]
                    rf_metrics.markdown(f"""
                    ### Random Forest Metrics:
                    - Accuracy: {rf_accuracy:.2f}
                    - Average time: {rf_avg_time*1000:.2f} ms
                    """)
                
                if metrics["CNN"]["total"] > 0:
                    cnn_accuracy = metrics["CNN"]["correct"] / metrics["CNN"]["total"]
                    cnn_avg_time = metrics["CNN"]["time"] / metrics["CNN"]["total"]
                    cnn_metrics.markdown(f"""
                    ### CNN Metrics:
                    - Accuracy: {cnn_accuracy:.2f}
                    - Average time: {cnn_avg_time*1000:.2f} ms
                    """)
                    
                if metrics["LSTM"]["total"] > 0:
                    lstm_accuracy = metrics["LSTM"]["correct"] / metrics["LSTM"]["total"]
                    lstm_avg_time = metrics["LSTM"]["time"] / metrics["LSTM"]["total"]
                    lstm_metrics.markdown(f"""
                    ### LSTM Metrics:
                    - Accuracy: {lstm_accuracy:.2f}
                    - Average time: {lstm_avg_time*1000:.2f} ms
                    """)
            
            # Обновление прогресс бара
            frame_count += 1
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            # Проверяем, была ли нажата кнопка остановки
            # Мы не создаем новую кнопку в цикле, а просто проверяем состояние
            if stop_pressed:
                break
            
        cap.release()
        os.unlink(tfile.name)  # Удаление временного файла
    
    # Итоговый вывод сравнения подходов
    if metrics["RF"]["total"] > 0 and metrics["CNN"]["total"] > 0 and metrics["LSTM"]["total"] > 0:
        rf_accuracy = metrics["RF"]["correct"] / metrics["RF"]["total"]
        rf_avg_time = metrics["RF"]["time"] / metrics["RF"]["total"]
        
        cnn_accuracy = metrics["CNN"]["correct"] / metrics["CNN"]["total"]
        cnn_avg_time = metrics["CNN"]["time"] / metrics["CNN"]["total"]
        
        lstm_accuracy = metrics["LSTM"]["correct"] / metrics["LSTM"]["total"]
        lstm_avg_time = metrics["LSTM"]["time"] / metrics["LSTM"]["total"]
        
        st.write("## Final Comparison of Approaches")
        
        # Добавляем дополнительные метрики для лучшего сравнения
        eval_mode_text = "Evaluation with human feedback" if evaluation_mode else "Evaluation with consensus voting"
        st.write(f"**Evaluation Method**: {eval_mode_text}")
        st.write(f"**Total Samples**: RF: {metrics['RF']['total']}, CNN: {metrics['CNN']['total']}, LSTM: {metrics['LSTM']['total']}")
        
        # Создаем датафрейм для сравнения
        comparison_data = {
            "Metric": ["Accuracy", "Average time (ms)", "Correct predictions", "Total predictions"],
            "Random Forest": [f"{rf_accuracy:.2f}", f"{rf_avg_time*1000:.2f}", 
                             f"{metrics['RF']['correct']}", f"{metrics['RF']['total']}"],
            "CNN": [f"{cnn_accuracy:.2f}", f"{cnn_avg_time*1000:.2f}", 
                   f"{metrics['CNN']['correct']}", f"{metrics['CNN']['total']}"],
            "LSTM": [f"{lstm_accuracy:.2f}", f"{lstm_avg_time*1000:.2f}", 
                    f"{metrics['LSTM']['correct']}", f"{metrics['LSTM']['total']}"]
        }
        
        st.table(comparison_data)
        
        # Выбор лучшего подхода по точности
        best_accuracy = max(rf_accuracy, cnn_accuracy, lstm_accuracy)
        if best_accuracy == rf_accuracy:
            st.write("**Accuracy Conclusion**: Random Forest showed the best accuracy.")
        elif best_accuracy == cnn_accuracy:
            st.write("**Accuracy Conclusion**: CNN showed the best accuracy.")
        else:
            st.write("**Accuracy Conclusion**: LSTM showed the best accuracy.")
            
        # Выбор лучшего подхода по скорости
        best_time = min(rf_avg_time, cnn_avg_time, lstm_avg_time)
        if best_time == rf_avg_time:
            st.write("**Speed Conclusion**: Random Forest is the fastest.")
        elif best_time == cnn_avg_time:
            st.write("**Speed Conclusion**: CNN is the fastest.")
        else:
            st.write("**Speed Conclusion**: LSTM is the fastest.")
        

# Боковая панель
st.sidebar.title("Settings")
source_option = st.sidebar.radio(
    "Select source",
    ["Webcam", "Upload video"]
)

# Режим оценки
st.sidebar.subheader("Evaluation Mode")
evaluation_mode = st.sidebar.checkbox("Enable Evaluation Mode", value=False, 
                                     help="In evaluation mode, you select the current gesture to measure model accuracy")
current_gesture = None
if evaluation_mode:
    current_gesture = st.sidebar.selectbox("Current Gesture", GESTURE_CLASSES)
    st.sidebar.info("Select the gesture you are currently showing to evaluate model accuracy")
else:
    st.sidebar.info("In automatic mode, model accuracy is calculated based on consensus voting between models")
    st.sidebar.warning("For more accurate evaluation, enable Evaluation Mode and select the gesture you are showing")

# Выбор подходов для сравнения
st.sidebar.subheader("Select approaches to compare")
compare_rf = st.sidebar.checkbox("Random Forest", value=True)
compare_cnn = st.sidebar.checkbox("CNN", value=True)
compare_lstm = st.sidebar.checkbox("LSTM", value=True)

if not (compare_rf or compare_cnn or compare_lstm):
    st.sidebar.warning("Please select at least one approach to compare")
    compare_rf = True

# Настройки стабилизации
st.sidebar.subheader("Stabilization Settings")
stability_window = st.sidebar.slider("Stability Window Size", 5, 30, 15, 
                                     help="Number of frames used for voting. Higher values make predictions more stable but slower to change.")
stability_threshold = st.sidebar.slider("Stability Threshold", 0.4, 0.9, 0.6, 0.05, 
                                        help="Confidence threshold for changing prediction. Higher values require more consistent predictions.")

# Создание кнопки перезапуска приложения
if st.sidebar.button("Restart Application"):
    st.experimental_rerun()

# Обработка выбора источника
if source_option == "Webcam":
    process_video(0)
else:
    uploaded_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        process_video(uploaded_file) 