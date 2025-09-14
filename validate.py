import os
import requests
import json
from tqdm import tqdm
import time
from PIL import Image


API_URL = "http://localhost:8000/detect"
VALIDATION_DIR = "./validation_data"
IOU_THRESHOLD = 0.5


def get_yolo_labels(label_path, img_width, img_height):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            _, x_center, y_center, width, height = [float(p) for p in parts]
            x_min = int((x_center - width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            x_max = int((x_center + width / 2) * img_width)
            y_max = int((y_center + height / 2) * img_height)
            boxes.append([x_min, y_min, x_max, y_max])
    return boxes


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou



def validate():
    image_dir = os.path.join(VALIDATION_DIR, "images")
    label_dir = os.path.join(VALIDATION_DIR, "labels")

    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        print(f"ОШИБКА: Папки 'images' или 'labels' не найдены внутри '{VALIDATION_DIR}'")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_processing_time = 0
    api_errors = 0

    print(f"Начинаем валидацию на {len(image_files)} изображениях...")

    for filename in tqdm(image_files):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

        try:
            with open(image_path, 'rb') as f:
                extension = filename.split('.')[-1].lower()
                if extension == 'jpg': extension = 'jpeg'  # Стандартный MIME-тип - image/jpeg
                content_type = f'image/{extension}'

                files = {'file': (filename, f, content_type)}

                start_time = time.time()
                response = requests.post(API_URL, files=files)
                total_processing_time += time.time() - start_time

                if response.status_code == 200:
                    pred_data = response.json().get('detections', [])
                    pred_boxes = [[d['bbox']['x_min'], d['bbox']['y_min'], d['bbox']['x_max'], d['bbox']['y_max']] for d
                                  in pred_data]
                else:
                    api_errors += 1
                    pred_boxes = []

        except requests.exceptions.ConnectionError:
            print("\nОШИБКА: Не удалось подключиться к API. Убедитесь, что Docker-контейнер запущен.")
            return
        except Exception as e:
            print(f"Ошибка при обработке файла {filename}: {e}")
            continue


        with Image.open(image_path) as img:
            w, h = img.size
        gt_boxes = get_yolo_labels(label_path, w, h)

        gt_boxes_matched = [False] * len(gt_boxes)
        pred_boxes_matched = [False] * len(pred_boxes)

        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                iou = calculate_iou(pred_box, gt_box)
                if iou >= IOU_THRESHOLD:
                    if not gt_boxes_matched[j]:
                        gt_boxes_matched[j] = True
                        pred_boxes_matched[i] = True
                        break

        true_positives += sum(pred_boxes_matched)
        false_positives += len(pred_boxes_matched) - sum(pred_boxes_matched)
        false_negatives += len(gt_boxes_matched) - sum(gt_boxes_matched)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_time = total_processing_time / len(image_files) if len(image_files) > 0 else 0

    print("\n" + "=" * 30)
    print("--- Результаты валидации ---")
    print(f"Всего изображений:     {len(image_files)}")
    print(f"Найдено совпадений (TP): {true_positives}")
    print(f"Ложных срабатываний (FP): {false_positives}")
    print(f"Пропущенных объектов (FN): {false_negatives}")
    if api_errors > 0:
        print(f"Ошибок API:             {api_errors}")
    print("-" * 30)
    print(f"Precision (Точность): {precision:.4f}")
    print(f"Recall (Полнота):    {recall:.4f}")
    print(f"F1-Score:             {f1_score:.4f}")
    print(f"Среднее время обработки: {avg_time:.4f} сек/изображение")
    print("=" * 30)


if __name__ == "__main__":
    validate()