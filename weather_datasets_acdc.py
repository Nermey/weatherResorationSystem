"""
weather_datasets.py — загрузчик ACDC (COCO JSON формат).

Используется val-сплит, т.к. test-аннотации в ACDC закрыты (только для лидерборда).
Val-выборка (~400 изображений на условие) полностью репрезентативна для оценки.

Структура:
    ACDC/images/fog/val/GOPR0475/frame_rgb_anon.png
    ACDC/labels/fog/instancesonly_fog_val_gt_detection.json

Cityscapes category_id → COMMON_ID:
    24/25 → person, 26/27 → car, 28 → bus, 32 → motorbike, 33 → bicycle
"""

import os, json
from collections import defaultdict
from PIL import Image
import torch

COMMON_CLASSES = ["person", "bicycle", "car", "motorbike", "bus"]
COMMON_ID      = {n: i for i, n in enumerate(COMMON_CLASSES)}

CITYSCAPES_TO_COMMON = {
    24: COMMON_ID["person"],
    25: COMMON_ID["person"],
    26: COMMON_ID["car"],
    27: COMMON_ID["car"],
    28: COMMON_ID["bus"],
    31: -1,
    32: COMMON_ID["motorbike"],
    33: COMMON_ID["bicycle"],
}


def _find_json(labels_root: str, weather: str) -> str:
    """
    Ищет JSON с bounding box аннотациями.
    Приоритет: 'detection' > 'gt' > всё остальное.
    Исключает 'image_info' (без аннотаций) и 'train'.
    """
    search_dirs = [
        os.path.join(labels_root, weather, "val"),
        os.path.join(labels_root, weather),
        labels_root,
    ]

    def priority(name):
        n = name.lower()
        if "image_info" in n: return -1
        if "train" in n:      return -1
        if "detection" in n:  return 3
        if "gt" in n:         return 2
        if "val" in n:        return 1
        return 0

    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        jsons = [f for f in os.listdir(d) if f.endswith(".json")]
        best = sorted(jsons, key=priority, reverse=True)
        if best and priority(best[0]) >= 0:
            return os.path.join(d, best[0])

    all_found = []
    for d in search_dirs:
        if os.path.isdir(d):
            all_found += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")]

    raise FileNotFoundError(
        f"JSON с аннотациями не найден для '{weather}'.\n"
        f"  Найденные JSON: {all_found}"
    )


def load_acdc(root: str, weather: str) -> "ACDCDataset":
    """
    Загружает val-сплит ACDC для одного типа погоды.

    root    — корневая папка (содержит images/ и labels/)
    weather — 'fog', 'rain' или 'snow'
    """
    images_root = os.path.join(root, "images")
    labels_root = os.path.join(root, "labels")

    json_path = _find_json(labels_root, weather)
    print(f"[ACDC/{weather}] JSON: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    if not data.get("annotations"):
        raise RuntimeError(
            f"В файле {json_path} нет поля 'annotations'!\n"
            f"  Это image_info файл. Нужен gt_detection файл.\n"
            f"  Запусти: ls {os.path.dirname(json_path)}"
        )

    id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}

    ann_by_image = defaultdict(list)
    for ann in data["annotations"]:
        cid    = ann["category_id"]
        common = CITYSCAPES_TO_COMMON.get(cid, -1)
        if common == -1:
            continue
        x, y, w, h = ann["bbox"]
        ann_by_image[ann["image_id"]].append([x, y, x + w, y + h, common])

    samples = []
    for img_id, fname in id_to_fname.items():
        if "/val/" not in fname:
            continue
        img_path = os.path.join(images_root, fname)
        if not os.path.exists(img_path):
            continue
        samples.append((img_path, ann_by_image.get(img_id, [])))

    if not samples:
        raise RuntimeError(
            f"Не найдено изображений для {weather}/val в {images_root}\n"
            f"  Проверь что папка images/{weather}/val/ существует."
        )

    print(f"[ACDC/{weather}] Загружено: {len(samples)} изображений")
    return ACDCDataset(samples)


class ACDCDataset:
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, anns = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if anns:
            boxes  = torch.tensor([a[:4] for a in anns], dtype=torch.float32)
            labels = torch.tensor([a[4]  for a in anns], dtype=torch.int64)
        else:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)

        return image, {"boxes": boxes, "labels": labels}, img_path