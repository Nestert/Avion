import cv2
import numpy as np
import mediapipe as mp
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Определение жестов
GESTURES = ["thumbs_up", "palm", "fist", "pointing", "victory"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def augment_sequence(sequence):
    """Применение случайных аугментаций к последовательности landmarks."""
    augmented_sequences = [sequence.copy()]  # Оригинальная последовательность
    seq_length, num_features = sequence.shape
    
    # 1. Добавление шума к координатам
    noise_factor = np.random.uniform(0.005, 0.02)
    noise = np.random.normal(0, noise_factor, sequence.shape)
    noisy_sequence = sequence + noise
    augmented_sequences.append(noisy_sequence)
    
    # 2. Масштабирование координат
    scale_factor = np.random.uniform(0.8, 1.2)
    # Предполагаем, что координаты x, y, z идут последовательно для каждой точки
    scaled_sequence = sequence.copy()
    # Масштабируем только x и y координаты (первые 2/3 признаков)
    for i in range(0, num_features, 3):
        scaled_sequence[:, i] = sequence[:, i] * scale_factor  # x
        scaled_sequence[:, i+1] = sequence[:, i+1] * scale_factor  # y
    augmented_sequences.append(scaled_sequence)
    
    # 3. Поворот координат в 2D плоскости
    angle = np.random.uniform(-15, 15) * (np.pi / 180.0)  # конвертируем в радианы
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_sequence = sequence.copy()
    for i in range(0, num_features, 3):
        xy_coords = sequence[:, i:i+2]  # x, y координаты
        rotated_xy = np.dot(xy_coords, rotation_matrix.T)
        rotated_sequence[:, i:i+2] = rotated_xy
    augmented_sequences.append(rotated_sequence)
    
    # 4. Перемещение (трансляция)
    translation_x = np.random.uniform(-0.1, 0.1)
    translation_y = np.random.uniform(-0.1, 0.1)
    translated_sequence = sequence.copy()
    for i in range(0, num_features, 3):
        translated_sequence[:, i] = sequence[:, i] + translation_x  # x
        translated_sequence[:, i+1] = sequence[:, i+1] + translation_y  # y
    augmented_sequences.append(translated_sequence)
    
    # 5. Зеркальное отражение по x
    flipped_sequence = sequence.copy()
    for i in range(0, num_features, 3):
        flipped_sequence[:, i] = 1.0 - sequence[:, i]  # инвертируем x координаты (предполагая, что они нормализованы от 0 до 1)
    augmented_sequences.append(flipped_sequence)
    
    # 6. Временное искажение (Time warping)
    # Создаем случайную функцию искажения времени
    if seq_length > 5:  # Минимальная длина для временного искажения
        time_indices = np.arange(seq_length)
        # Создаем новые индексы времени с небольшими искажениями
        warped_indices = np.linspace(0, seq_length-1, seq_length)
        warped_indices += np.random.normal(0, 0.5, seq_length)
        warped_indices = np.clip(warped_indices, 0, seq_length-1)
        
        # Интерполируем последовательность по новым индексам
        warped_sequence = np.zeros_like(sequence)
        for i in range(num_features):
            warped_sequence[:, i] = np.interp(warped_indices, time_indices, sequence[:, i])
        augmented_sequences.append(warped_sequence)
    
    # 7. Случайный дропаут кадров
    if seq_length > 3:  # Минимальная длина для дропаута
        dropout_sequence = sequence.copy()
        # Выбираем случайные индексы для дропаута (не более 20% кадров)
        num_dropout = int(seq_length * np.random.uniform(0.05, 0.2))
        dropout_indices = np.random.choice(seq_length, num_dropout, replace=False)
        
        # Заменяем выбранные кадры средним значением соседних кадров
        for idx in dropout_indices:
            if idx > 0 and idx < seq_length - 1:
                dropout_sequence[idx] = (sequence[idx-1] + sequence[idx+1]) / 2
            elif idx > 0:
                dropout_sequence[idx] = sequence[idx-1]
            else:
                dropout_sequence[idx] = sequence[idx+1]
        augmented_sequences.append(dropout_sequence)
    
    # 8. Изменение скорости (ускорение/замедление)
    speed_factor = np.random.uniform(0.8, 1.2)
    new_length = int(seq_length * speed_factor)
    if new_length > 2:  # Минимальная длина
        # Интерполируем к новой длине, затем обратно к исходной для сохранения размерности
        new_indices = np.linspace(0, seq_length-1, new_length)
        speed_sequence = np.zeros((new_length, num_features))
        for i in range(num_features):
            speed_sequence[:, i] = np.interp(new_indices, np.arange(seq_length), sequence[:, i])
        
        # Теперь интерполируем обратно к исходной длине
        original_indices = np.linspace(0, new_length-1, seq_length)
        resampled_sequence = np.zeros_like(sequence)
        for i in range(num_features):
            resampled_sequence[:, i] = np.interp(original_indices, np.arange(new_length), speed_sequence[:, i])
        augmented_sequences.append(resampled_sequence)
    
    return augmented_sequences

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

def collect_sequence_data(image_paths, sequence_length=15, use_augmentation=True):
    """Сбор последовательностей данных из изображений с применением аугментаций."""
    sequences = []
    labels = []
    
    for label_idx, label in enumerate(GESTURES):
        label_dir = os.path.join(image_paths, label)
        
        if not os.path.exists(label_dir):
            print(f"Директория {label_dir} не найдена. Пропускаем.")
            continue
        
        print(f"Обработка изображений для жеста: {label}")
        
        # Собираем все изображения в директории
        image_files = [f for f in os.listdir(label_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Сортируем изображения по имени для сохранения последовательности
        image_files.sort(key=lambda x: os.path.getmtime(os.path.join(label_dir, x)))
        
        # Обработка изображений и создание последовательностей
        all_landmarks = []
        
        for image_file in image_files:
            image_path = os.path.join(label_dir, image_file)
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"Не удалось прочитать изображение: {image_path}")
                continue
                
            landmarks, _ = extract_hand_landmarks(frame)
            
            if landmarks:
                all_landmarks.append(landmarks)
        
        # Создаем последовательности из полученных ориентиров
        if len(all_landmarks) >= sequence_length:
            # Создаем последовательности с перекрытием
            for i in range(0, len(all_landmarks) - sequence_length + 1, 5):  # Шаг 5 для перекрытия
                sequence = all_landmarks[i:i + sequence_length]
                sequences.append(sequence)
                labels.append(label_idx)
                
                # Применение аугментаций, если включено
                if use_augmentation:
                    sequence_array = np.array(sequence)
                    augmented_sequences = augment_sequence(sequence_array)
                    
                    # Пропускаем первый элемент, так как это оригинальная последовательность
                    for aug_seq in augmented_sequences[1:]:
                        sequences.append(aug_seq)
                        labels.append(label_idx)
                
            print(f"  Создано {len(sequences) - sum(1 for l in labels if l != label_idx)} последовательностей для жеста {label}")
        else:
            print(f"  Недостаточно данных для создания последовательности. Необходимо {sequence_length}, найдено {len(all_landmarks)}")
    
    return np.array(sequences), np.array(labels)

# Определение LSTM модели с использованием PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
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

# Класс для ранней остановки
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping счетчик: {self.counter} из {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Валидационные потери уменьшились ({self.val_loss_min:.6f} --> {val_loss:.6f}). Сохранение модели...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def normalize_sequences(sequences):
    """Нормализация последовательностей данных."""
    # Получаем размерности
    num_sequences, seq_length, num_features = sequences.shape
    
    # Преобразуем 3D массив в 2D для нормализации
    sequences_reshaped = sequences.reshape(num_sequences * seq_length, num_features)
    
    # Применяем нормализацию
    scaler = StandardScaler()
    sequences_normalized = scaler.fit_transform(sequences_reshaped)
    
    # Преобразуем обратно в 3D форму
    sequences_normalized = sequences_normalized.reshape(num_sequences, seq_length, num_features)
    
    return sequences_normalized, scaler

def train_lstm_model(sequences, labels):
    """Обучение LSTM модели на последовательностях данных с использованием PyTorch."""
    # Нормализация данных
    sequences_normalized, scaler = normalize_sequences(sequences)
    
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        sequences_normalized, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Преобразование в тензоры PyTorch
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Создание датасетов и загрузчиков данных
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Определение устройства (CPU или GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создание модели
    input_size = X_train.shape[2]  
    hidden_size = 128
    num_layers = 1
    num_classes = len(np.unique(labels))
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=0.3
    ).to(device)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Шедулер скорости обучения
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                verbose=True, min_lr=1e-6)
    
    # Ранняя остановка
    early_stopping = EarlyStopping(patience=15, verbose=True, path='models/lstm_checkpoint.pt')
    
    # Для отслеживания метрик обучения
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Обучение модели
    start_time = time.time()
    num_epochs = 50  
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Градиентный клиппинг для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Валидация
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        # Применение шедулера
        scheduler.step(val_epoch_loss)
        
        # Проверка ранней остановки
        early_stopping(val_epoch_loss, model)
        
        print(f'Эпоха {epoch+1}/{num_epochs} | '
              f'Потери: {epoch_loss:.4f} | '
              f'Точность: {epoch_acc:.4f} | '
              f'Вал. потери: {val_epoch_loss:.4f} | '
              f'Вал. точность: {val_epoch_acc:.4f}')
        
        if early_stopping.early_stop:
            print("Ранняя остановка!")
            break
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load('models/lstm_checkpoint.pt'))
    
    training_time = time.time() - start_time
    
    # Оценка модели
    model.eval()
    start_time = time.time()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    inference_time = (time.time() - start_time) / len(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    report = classification_report(all_targets, all_preds)
    
    # Создание объекта "history" аналогичного Keras для совместимости
    class HistoryObject:
        def __init__(self, train_accs, val_accs, train_losses, val_losses):
            self.history = {
                'accuracy': train_accs,
                'val_accuracy': val_accs,
                'loss': train_losses,
                'val_loss': val_losses
            }
    
    history = HistoryObject(train_accs, val_accs, train_losses, val_losses)
    
    return model, history, accuracy, report, training_time, inference_time, scaler

def simulate_sequence_data(num_sequences=100, sequence_length=15):
    """Симуляция данных последовательностей для демонстрации."""
    print("Симуляция последовательностей данных для LSTM...")
    
    # Количество признаков (x, y, z для 21 точки руки = 63 признака)
    num_features = 63
    
    # Создание случайных последовательностей с более четкой структурой
    sequences = []
    labels = []
    
    for i in range(num_sequences):
        # Определяем класс
        label = i % len(GESTURES)
        labels.append(label)
        
        # Создаем базовый шаблон для класса
        base_pattern = np.random.randn(1, num_features) * 0.1 + label * 0.2
        
        # Создаем последовательность с базовым шаблоном и шумом
        sequence = []
        for j in range(sequence_length):
            # Добавляем тренд, зависящий от времени
            time_factor = j / sequence_length
            frame = base_pattern + np.random.randn(1, num_features) * 0.05 + time_factor * 0.1
            sequence.append(frame[0])
        
        sequences.append(sequence)
    
    return np.array(sequences), np.array(labels)

def main():
    """Основная функция для обучения LSTM модели."""
    image_dir = "gesture_data"
    
    # Создание папки для результатов обучения и моделей, если их нет
    os.makedirs("training_results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Проверка наличия реальных данных
    if os.path.exists(image_dir):
        print("Сбор данных из изображений с применением аугментаций...")
        sequences, labels = collect_sequence_data(image_dir, use_augmentation=True)
    else:
        print(f"Директория {image_dir} не найдена. Используем симулированные данные.")
        sequences, labels = simulate_sequence_data(num_sequences=500)  # Увеличиваем количество симулируемых данных
    
    if len(sequences) == 0:
        print("Не удалось собрать последовательности данных.")
        return
    
    print(f"Собрано {len(sequences)} последовательностей для обучения LSTM (с аугментациями).")
    
    # Обучение LSTM модели
    print("\nОбучение модели LSTM...")
    lstm_model, lstm_history, lstm_accuracy, lstm_report, lstm_time, lstm_inference, scaler = train_lstm_model(sequences, labels)
    
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
    plt.savefig("training_results/lstm_training_results.png")
    plt.close()
    
    # Сохранение модели и скейлера
    print("\nСохранение модели LSTM...")
    torch.save(lstm_model.state_dict(), "models/lstm_model.pth")
    # Сохраняем скейлер для последующего использования при инференсе
    import pickle
    with open('models/lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Сохранение результатов в CSV и создание сводной таблицы
    results = {
        "Метрика": ["Точность", "Время обучения (с)", "Время инференса (мс)"],
        "LSTM": [f"{lstm_accuracy:.4f}", f"{lstm_time:.2f}", f"{lstm_inference*1000:.2f}"]
    }
    
    results_df = pd.DataFrame(results)
    print("\nРезультаты LSTM модели:")
    print(results_df)
    results_df.to_csv("training_results/lstm_results.csv", index=False)
    
    print("\nОбучение LSTM модели завершено. Результаты сохранены.")

if __name__ == "__main__":
    main() 