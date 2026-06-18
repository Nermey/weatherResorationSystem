"""
weather_datasets.py
===================
Загрузчик датасета DAWN в YOLO txt формате.

Структура:
    images/
        fog_001.jpg (или любой паттерн с разделением по погоде в названии)
        rain_002.jpg
        snow_003.jpg
        ...
    labels/
        fog_001.txt
        rain_002.txt
        snow_003.txt
        ...

Формат .txt (YOLO):
    class_id center_x center_y width height
    (координаты нормализованы, 0-1)
"""

import os
import glob
import re
from pathlib import Path
from PIL import Image
import torch

# ── Классы ───────────────────────────────────────────────────────────────────

COMMON_CLASSES = ["person", "bicycle", "car", "motorbike", "bus"]
COMMON_ID = {name: i for i, name in enumerate(COMMON_CLASSES)}

# Маппинг ID класса в COMMON_ID. Адаптируй под твоих ID!
# Пример: если в твоём датасете класс 3 = машина, класс 1 = человек
YOLO_CLASS_MAP = {
    0: COMMON_ID.get("person", -1),      # если 0 = person
    1: COMMON_ID.get("person", -1),      # если 1 = person
    3: COMMON_ID.get("car", -1),         # если 3 = car
    6: COMMON_ID.get("car", -1),         # если 6 = car (грузовик)
    # Добавь свои маппинги по необходимости
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")

# Паттерны для определения типа погоды из имени файла
WEATHER_PATTERNS = {
    "fogsmog": ["fog", "fogsmog", "haze"],  # приоритет: сначала fogsmog
    "rain":    ["rain"],
    "snow":    ["snow"],
}


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _detect_weather_from_filename(filename: str) -> str | None:
    """
    Определяет тип погоды из имени файла.
    Пример: "fog_001.jpg" → "fogsmog"
    """
    name_lower = filename.lower()

    # Проверяем в порядке приоритета
    for weather, keywords in WEATHER_PATTERNS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return weather

    return None


def _yolo_to_xyxy(cx_norm: float, cy_norm: float, w_norm: float, h_norm: float,
                  img_w: int, img_h: int) -> tuple:
    """
    Конвертирует YOLO формат (нормализованные center_x, center_y, width, height)
    в xyxy формат (xmin, ymin, xmax, ymax) в пиксельных координатах.
    """
    # Денормализуем
    cx = cx_norm * img_w
    cy = cy_norm * img_h
    w = w_norm * img_w
    h = h_norm * img_h

    # Конвертируем в xyxy
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2

    return (xmin, ymin, xmax, ymax)


# ── Основной класс датасета ──────────────────────────────────────────────────

class YOLOWeatherDataset:
    """
    Датасет с YOLO txt аннотациями.

    Загружает изображения и соответствующие .txt файлы.
    Определяет тип погоды из имени файла.
    """

    def __init__(self, images_dir: str, labels_dir: str, weather_types: list | None = None):
        """
        Args:
            images_dir: папка с изображениями
            labels_dir: папка с .txt файлами (YOLO формат)
            weather_types: список типов погоды для фильтрации
                          (например, ["fogsmog", "rain", "snow"])
                          Если None — загружает всё
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.samples = []

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Папка images не найдена: {images_dir}")
        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"Папка labels не найдена: {labels_dir}")

        # Ищем пары изображение-разметка
        for img_path in sorted(glob.glob(os.path.join(images_dir, "*.*"))):
            # Проверяем расширение
            if not any(img_path.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                continue

            # Соответствующий .txt файл
            stem = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(labels_dir, stem + ".txt")

            if not os.path.exists(txt_path):
                continue

            # Определяем тип погоды
            weather = _detect_weather_from_filename(os.path.basename(img_path))

            # Фильтруем по типу погоды если задан
            if weather_types is not None and weather not in weather_types:
                continue

            self.samples.append((img_path, txt_path, weather))

        if not self.samples:
            filter_str = f" (фильтр: {weather_types})" if weather_types else ""
            raise RuntimeError(
                f"Не найдено пар изображение-разметка{filter_str}\n"
                f"  Изображения: {images_dir}\n"
                f"  Разметка:    {labels_dir}\n"
                f"  Пример: поместите fog_001.jpg и fog_001.txt рядом друг с другом."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path, weather = self.samples[idx]

        # Загружаем изображение
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Загружаем аннотации
        boxes, labels = [], []
        try:
            with open(txt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    try:
                        class_id = int(parts[0])
                        cx_norm = float(parts[1])
                        cy_norm = float(parts[2])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                    except ValueError:
                        continue

                    # Маппим класс
                    if class_id not in YOLO_CLASS_MAP:
                        continue
                    common_label = YOLO_CLASS_MAP[class_id]
                    if common_label == -1:  # класс не в маппинге
                        continue

                    # Конвертируем координаты
                    xmin, ymin, xmax, ymax = _yolo_to_xyxy(
                        cx_norm, cy_norm, w_norm, h_norm, img_w, img_h
                    )
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(common_label)

        except Exception as e:
            print(f"  ⚠ Ошибка при чтении {txt_path}: {e}")

        target = {
            "boxes":  torch.tensor(boxes,  dtype=torch.float32).reshape(-1, 4) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }

        return image, target, img_path


# ── Публичные функции ────────────────────────────────────────────────────────

def load_dawn(root_dir: str, weather_type: str | None = None) -> YOLOWeatherDataset:
    """
    Загружает датасет DAWN в формате YOLO.

    Args:
        root_dir: путь к папке, содержащей images/ и labels/
        weather_type: конкретный тип погоды ('fogsmog', 'rain', 'snow')
                      Если None — загружает все типы

    Пример структуры:
        data/DAWN/
            images/
                fog_001.jpg
                rain_002.jpg
                snow_003.jpg
            labels/
                fog_001.txt
                rain_002.txt
                snow_003.txt
    """
    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")

    weather_types = [weather_type] if weather_type else None

    print(f"[DAWN] Формат: YOLO txt | Путь: {root_dir}")
    if weather_type:
        print(f"        Фильтр: {weather_type}")

    dataset = YOLOWeatherDataset(images_dir, labels_dir, weather_types=weather_types)

    print(f"[DAWN] Загружено {len(dataset)} изображений")
    return dataset