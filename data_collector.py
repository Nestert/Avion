import cv2
import numpy as np
import mediapipe as mp
import os
import time
import argparse

# Определение жестов
GESTURES = ["thumbs_up", "palm", "fist", "pointing", "victory"]

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def collect_data(output_dir, gesture_index, num_samples=150):
    """Сбор данных с веб-камеры для указанного жеста."""
    # Проверка валидности индекса жеста
    if gesture_index < 0 or gesture_index >= len(GESTURES):
        print(f"Ошибка: индекс жеста должен быть от 0 до {len(GESTURES) - 1}")
        return
    
    gesture = GESTURES[gesture_index]
    print(f"Сбор данных для жеста: {gesture}")
    
    # Создание директории для сохранения изображений
    gesture_dir = os.path.join(output_dir, gesture)
    os.makedirs(gesture_dir, exist_ok=True)
    
    # Проверка количества существующих образцов
    existing_samples = len([f for f in os.listdir(gesture_dir) if f.endswith('.jpg')])
    print(f"Найдено {existing_samples} существующих образцов")
    
    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    
    collected_samples = 0
    delay_counter = 0
    sample_interval = 5  # Интервал между захватом образцов (в кадрах)
    
    # Строка инструкций
    instruction_text = f"Сбор данных для жеста: {gesture.upper()} ({GESTURES.index(gesture) + 1}/{len(GESTURES)})"
    
    # Строка описаний жестов
    gesture_descriptions = {
        "thumbs_up": "Большой палец вверх",
        "palm": "Открытая ладонь",
        "fist": "Сжатый кулак",
        "pointing": "Указательный палец",
        "victory": "Знак победы (V)"
    }
    
    description_text = f"Покажите жест: {gesture_descriptions.get(gesture, gesture)}"
    
    # Счетчик для отображения обратного отсчета
    countdown = 3
    countdown_start = time.time()
    collecting = False
    
    print("Подготовьтесь...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Отражение по горизонтали для естественного отображения
        frame = cv2.flip(frame, 1)
        
        # Отображение инструкций
        cv2.putText(frame, instruction_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, description_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Отображение прогресса
        progress_text = f"Собрано: {collected_samples + existing_samples}/{num_samples + existing_samples}"
        cv2.putText(frame, progress_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Обработка кадра для обнаружения рук
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Проверка наличия руки в кадре
        if results.multi_hand_landmarks:
            # Отрисовка найденных рук
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
            
            # Если не начали сбор, показываем обратный отсчет
            current_time = time.time()
            if not collecting:
                elapsed = current_time - countdown_start
                remaining = max(0, 3 - int(elapsed))
                
                if remaining > 0:
                    cv2.putText(frame, f"Начало через: {remaining}", (frame.shape[1] // 2 - 100, frame.shape[0] // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                else:
                    collecting = True
                    print("Начинаем сбор данных...")
            
            # Если начали сбор, сохраняем образцы
            if collecting:
                # Сохранение образца с интервалом
                if delay_counter % sample_interval == 0 and collected_samples < num_samples:
                    timestamp = int(time.time() * 1000)
                    file_path = os.path.join(gesture_dir, f"{gesture}_{timestamp}.jpg")
                    cv2.imwrite(file_path, frame)
                    collected_samples += 1
                    print(f"Сохранен образец {collected_samples}/{num_samples}")
                
                delay_counter += 1
            
        else:
            # Если рука не обнаружена, сбрасываем счетчик обратного отсчета
            if not collecting:
                countdown_start = time.time()
            cv2.putText(frame, "Рука не обнаружена!", (frame.shape[1] // 2 - 150, frame.shape[0] // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Отображение текущего кадра
        cv2.imshow("Сбор данных для жестов", frame)
        
        # Проверка завершения сбора
        if collected_samples >= num_samples:
            print(f"Сбор данных для жеста {gesture} завершен!")
            break
        
        # Выход при нажатии клавиши 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return collected_samples

def main():
    """Основная функция программы."""
    parser = argparse.ArgumentParser(description='Сбор данных для распознавания жестов')
    parser.add_argument('--output', type=str, default='gesture_data',
                        help='Директория для сохранения данных')
    parser.add_argument('--gesture', type=int, default=-1,
                        help=f'Индекс жеста для сбора (0-{len(GESTURES)-1}), или -1 для сбора всех жестов')
    parser.add_argument('--samples', type=int, default=150,
                        help='Количество образцов для сбора каждого жеста')
    
    args = parser.parse_args()
    
    # Создание основной директории
    os.makedirs(args.output, exist_ok=True)
    
    print("Программа сбора данных для распознавания жестов")
    print(f"Доступные жесты: {', '.join([f'{i}: {g}' for i, g in enumerate(GESTURES)])}")
    
    if args.gesture >= 0:
        # Сбор данных для одного жеста
        collect_data(args.output, args.gesture, args.samples)
    else:
        # Сбор данных для всех жестов по порядку
        for i, gesture in enumerate(GESTURES):
            print(f"\n{'=' * 30}")
            print(f"Жест {i+1}/{len(GESTURES)}: {gesture}")
            print(f"{'=' * 30}\n")
            
            collect_data(args.output, i, args.samples)
            
            # Пауза между жестами
            if i < len(GESTURES) - 1:
                print("Нажмите любую клавишу для перехода к следующему жесту...")
                cv2.waitKey(0)
    
    print("\nСбор данных завершен!")
    print(f"Собранные данные сохранены в директории: {args.output}")

if __name__ == "__main__":
    main() 