import cv2
import numpy as np
import mediapipe as mp
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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

def augment_image(image):
    """Применение случайных аугментаций к изображению."""
    augmented_images = [image]  # Оригинальное изображение тоже сохраняем
    h, w = image.shape[:2]
    
    # 1. Поворот на случайный угол
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated)
    
    # 2. Изменение масштаба
    scale = np.random.uniform(0.8, 1.2)
    scaled = cv2.resize(image, None, fx=scale, fy=scale)
    scaled = cv2.resize(scaled, (w, h))  # Возвращаем к исходному размеру
    augmented_images.append(scaled)
    
    # 3. Изменение яркости
    brightness = np.random.uniform(0.7, 1.3)
    bright = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    augmented_images.append(bright)
    
    # 4. Добавление гауссовского шума
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented_images.append(noisy)
    
    # 5. Горизонтальное отражение
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # 6. Размытие изображения
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmented_images.append(blurred)
    
    # 7. Случайное обрезание и масштабирование
    crop_factor = np.random.uniform(0.8, 0.95)
    crop_h, crop_w = int(h * crop_factor), int(w * crop_factor)
    start_h = np.random.randint(0, h - crop_h + 1)
    start_w = np.random.randint(0, w - crop_w + 1)
    cropped = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
    cropped = cv2.resize(cropped, (w, h))
    augmented_images.append(cropped)
    
    # 8. Изменение контраста
    contrast = np.random.uniform(0.7, 1.3)
    mean = np.mean(image)
    contrasted = (image - mean) * contrast + mean
    contrasted = np.clip(contrasted, 0, 255).astype(np.uint8)
    augmented_images.append(contrasted)
    
    return augmented_images

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

def extract_features_from_augmented(image, image_rgb=None):
    """Извлечение признаков из уже аугментированного изображения."""
    if image_rgb is None:
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

def collect_dataset(data_dir, use_augmentation=True):
    """Сбор данных из каталогов с изображениями с применением аугментаций."""
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
                hand_crop_norm = hand_crop / 255.0
                image_data.append(hand_crop_norm)
                image_labels.append(gesture)
                
                # Применение аугментаций, если включено
                if use_augmentation:
                    augmented_images = augment_image(hand_crop)
                    
                    # Пропускаем первое изображение, так как это оригинал
                    for aug_img in augmented_images[1:]:
                        # Получаем признаки из аугментированного изображения
                        aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                        aug_features, aug_crop = extract_features_from_augmented(aug_img, aug_img_rgb)
                        
                        if aug_features is not None and aug_crop is not None:
                            features_data.append(aug_features)
                            labels.append(gesture)
                            
                            # Нормализация для CNN
                            aug_crop_norm = aug_crop / 255.0
                            image_data.append(aug_crop_norm)
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

# Определение CNN модели с использованием PyTorch
class CNNModel(nn.Module):
    def __init__(self, num_classes):
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

def train_cnn(images, labels):
    """Обучение сверточной нейронной сети с использованием PyTorch."""
    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
    
    # Подготовка данных для PyTorch (изменение порядка осей для каналов)
    X_train = np.transpose(X_train, (0, 3, 1, 2))
    X_test = np.transpose(X_test, (0, 3, 1, 2))
    
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
    num_classes = len(np.unique(labels))
    model = CNNModel(num_classes).to(device)
    
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
    plt.savefig("training_results/results_comparison.png")
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
    
    # Создание папки для результатов обучения, если её нет
    os.makedirs("training_results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("Сбор данных из изображений с применением аугментаций...")
    features_data, encoded_labels, image_data, encoded_image_labels, label_encoder = collect_dataset(data_dir, use_augmentation=True)
    
    if len(features_data) == 0 or len(image_data) == 0:
        print("Не удалось собрать данные. Проверьте наличие изображений в папках.")
        return
    
    print(f"Собрано {len(features_data)} образцов для обучения Random Forest (с аугментациями).")
    print(f"Собрано {len(image_data)} образцов для обучения CNN (с аугментациями).")
    
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
    with open("models/random_forest_model.pkl", "wb") as f:
        pickle.dump((rf_model, label_encoder), f)
    
    # Сохранение PyTorch модели
    torch.save(cnn_model.state_dict(), "models/cnn_model.pth")
    
    # Сохранение информации о классах
    with open("models/class_info.pkl", "wb") as f:
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
    results_df.to_csv("training_results/results_comparison.csv", index=False)
    
    print("\nОбучение моделей завершено. Результаты сохранены.")

if __name__ == "__main__":
    main() 