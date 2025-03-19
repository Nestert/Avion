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

def collect_sequence_data(image_paths, sequence_length=15):
    """Сбор последовательностей данных из изображений."""
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
    input_size = X_train.shape[2]  # Количество признаков
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
    num_epochs = 50  # Увеличиваем потенциальное число эпох, т.к. ранняя остановка предотвратит переобучение
    
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
        print("Сбор данных из изображений...")
        sequences, labels = collect_sequence_data(image_dir)
    else:
        print(f"Директория {image_dir} не найдена. Используем симулированные данные.")
        sequences, labels = simulate_sequence_data(num_sequences=500)  # Увеличиваем количество симулируемых данных
    
    if len(sequences) == 0:
        print("Не удалось собрать последовательности данных.")
        return
    
    print(f"Собрано {len(sequences)} последовательностей для обучения LSTM.")
    
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