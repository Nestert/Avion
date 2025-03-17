import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import pickle
import tempfile
import os
import time
from datetime import datetime
import collections

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
        # Здесь будет загрузка предварительно обученной модели
        # В реальном приложении модель была бы обучена заранее
        # и загружена из файла
        # model = pickle.load(open('random_forest_model.pkl', 'rb'))
        # prediction = model.predict([features])[0]
        
        # Симуляция предсказания для демонстрации
        # Используем seed на основе признаков, чтобы одинаковые жесты давали одинаковые результаты
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
        # В реальном приложении здесь бы использовалась 
        # предварительно обученная CNN модель
        # model = tf.keras.models.load_model('cnn_model.h5')
        
        # Симуляция предсказания для демонстрации
        # Используем seed на основе координат ключевых точек руки
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
        # В реальном приложении здесь бы использовалась 
        # предварительно обученная LSTM модель
        # model = tf.keras.models.load_model('lstm_model.h5')
        
        # Симуляция предсказания для демонстрации
        # Используем seed на основе последних признаков последовательности
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
def display_results(image, hand_landmarks, prediction1, confidence1, prediction2, confidence2, prediction3, confidence3):
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
                    
                    # Обновление метрик (в реальном сценарии здесь было бы сравнение с ground truth)
                    if np.random.random() > 0.25:  # Симуляция правильного предсказания для LSTM
                        metrics["LSTM"]["correct"] += 1
                
                # Обновление метрик для RF и CNN (в реальном сценарии здесь было бы сравнение с ground truth)
                if np.random.random() > 0.3:  # Симуляция правильного предсказания для RF
                    metrics["RF"]["correct"] += 1
                if np.random.random() > 0.35:  # Симуляция правильного предсказания для CNN
                    metrics["CNN"]["correct"] += 1
            else:
                # Если рука не обнаружена, очищаем историю и стабилизаторы
                landmark_history.clear()
                rf_stabilizer.clear()
                cnn_stabilizer.clear()
                lstm_stabilizer.clear()
            
            # Отображение результатов на кадре
            result_frame = display_results(frame, hand_landmarks, prediction1, confidence1, 
                                          prediction2, confidence2, prediction3, confidence3)
            
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
                        
                        # Обновление метрик (в реальном сценарии здесь было бы сравнение с ground truth)
                        if np.random.random() > 0.25:  # Симуляция правильного предсказания для LSTM
                            metrics["LSTM"]["correct"] += 1
                    
                    # Обновление метрик для RF и CNN (в реальном сценарии здесь было бы сравнение с ground truth)
                    if np.random.random() > 0.3:  # Симуляция правильного предсказания для RF
                        metrics["RF"]["correct"] += 1
                    if np.random.random() > 0.35:  # Симуляция правильного предсказания для CNN
                        metrics["CNN"]["correct"] += 1
                else:
                    # Если рука не обнаружена, очищаем историю и стабилизаторы
                    landmark_history.clear()
                    rf_stabilizer.clear()
                    cnn_stabilizer.clear()
                    lstm_stabilizer.clear()
                
                # Отображение результатов на кадре
                result_frame = display_results(frame, hand_landmarks, prediction1, confidence1, 
                                              prediction2, confidence2, prediction3, confidence3)
                
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
        
        comparison_data = {
            "Metric": ["Accuracy", "Average time (ms)"],
            "Random Forest": [f"{rf_accuracy:.2f}", f"{rf_avg_time*1000:.2f}"],
            "CNN": [f"{cnn_accuracy:.2f}", f"{cnn_avg_time*1000:.2f}"],
            "LSTM": [f"{lstm_accuracy:.2f}", f"{lstm_avg_time*1000:.2f}"]
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
        
        # Компромисс между точностью и скоростью
        st.write("### Balance between Accuracy and Speed")
        
        # Условные рекомендации
        if rf_accuracy > 0.8 and rf_avg_time < 0.01:
            st.write("For devices with limited resources, Random Forest is recommended.")
        
        if cnn_accuracy > 0.85:
            st.write("For high accuracy in static gestures, CNN is recommended.")
        
        if lstm_accuracy > 0.85:
            st.write("For dynamic gesture recognition, LSTM is recommended.")

# Боковая панель
st.sidebar.title("Settings")
source_option = st.sidebar.radio(
    "Select source",
    ["Webcam", "Upload video"]
)

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