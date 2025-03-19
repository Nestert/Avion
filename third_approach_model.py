import cv2
import numpy as np
import mediapipe as mp
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

# Определение LSTM модели с использованием PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM слои
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, 
                            batch_first=True, bidirectional=False)
        self.dropout1 = nn.Dropout(0.2)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size*2, num_layers=1, 
                            batch_first=True, bidirectional=False)
        self.dropout2 = nn.Dropout(0.2)
        
        self.lstm3 = nn.LSTM(hidden_size*2, hidden_size, num_layers=1, 
                            batch_first=True, bidirectional=False)
        
        # Полносвязные слои для классификации
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Первый LSTM слой
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Второй LSTM слой
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        
        # Третий LSTM слой
        out, _ = self.lstm3(out)
        
        # Берем только выход последнего временного шага
        out = out[:, -1, :]
        
        # Полносвязные слои
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out

def train_lstm_model(sequences, labels):
    """Обучение LSTM модели на последовательностях данных с использованием PyTorch."""
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.3, random_state=42
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
    hidden_size = 64
    num_layers = 1
    num_classes = len(np.unique(labels))
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)
    
    # Определение функции потерь и оптимизатора
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Для отслеживания метрик обучения
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Обучение модели
    start_time = time.time()
    num_epochs = 30
    
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
        
        print(f'Эпоха {epoch+1}/{num_epochs} | '
              f'Потери: {epoch_loss:.4f} | '
              f'Точность: {epoch_acc:.4f} | '
              f'Вал. потери: {val_epoch_loss:.4f} | '
              f'Вал. точность: {val_epoch_acc:.4f}')
    
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
    
    # Создание папки для результатов обучения, если её нет
    os.makedirs("training_results", exist_ok=True)
    
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
    plt.savefig("training_results/lstm_training_results.png")
    plt.close()
    
    # Сохранение модели
    print("\nСохранение модели LSTM...")
    torch.save(lstm_model.state_dict(), "models/lstm_model.pth")
    
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