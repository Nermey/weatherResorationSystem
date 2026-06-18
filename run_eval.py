"""
run_eval.py
===========
Сравнение детекции объектов в двух режимах:
  - baseline:   исходное погодное изображение → детектор
  - с системой: изображение → твоя система → детектор

Датасет: DAWN в YOLO txt формате (images/ и labels/)
Детекторы: YOLOv8s, Faster R-CNN ResNet50

Структура:
    data/DAWN/
        images/      ← все картинки (fog_001.jpg, rain_002.jpg, snow_003.jpg, ...)
        labels/      ← все .txt файлы (fog_001.txt, rain_002.txt, snow_003.txt, ...)

Зависимости:
    pip install ultralytics torchvision torchmetrics pillow
"""

import ssl
import torch
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO
# from torchvision.models.detection import (
#     fasterrcnn_resnet50_fpn,
#     FasterRCNN_ResNet50_FPN_Weights,
# )
from PIL import Image

from weather_datasets import load_dawn, COMMON_ID, COMMON_CLASSES
from main import WeatherRestorationSystem

# ── Исправление SSL на macOS ─────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context

# ── Маппинг классов COCO ─────────────────────────────────────────────────────

YOLO_COCO = {
    "person":    0,
    "bicycle":   1,
    "car":       2,
    "motorbike": 3,
    "bus":       5,
}

FRCNN_COCO = {
    "person":    1,
    "bicycle":   2,
    "car":       3,
    "motorbike": 4,
    "bus":       6,
}

YOLO_TO_COMMON  = {YOLO_COCO[n]:  COMMON_ID[n] for n in COMMON_CLASSES}
FRCNN_TO_COMMON = {FRCNN_COCO[n]: COMMON_ID[n] for n in COMMON_CLASSES}


def _empty_pred():
    return {
        "boxes":  torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,),   dtype=torch.float32),
        "labels": torch.zeros((0,),   dtype=torch.int64),
    }


# ── YOLOv8 ───────────────────────────────────────────────────────────────────

def load_yolo() -> YOLO:
    return YOLO("yolo26s.pt")


def predict_yolo(model: YOLO, image: Image.Image, device: str) -> dict:
    results = model.predict(image, device=device, verbose=False)[0]
    boxes_tensor = results.boxes
    if boxes_tensor is None or len(boxes_tensor) == 0:
        return _empty_pred()

    keep_boxes, keep_scores, keep_labels = [], [], []
    for xyxy, conf, cls in zip(
        boxes_tensor.xyxy, boxes_tensor.conf, boxes_tensor.cls
    ):
        coco_id = int(cls.item())
        if coco_id in YOLO_TO_COMMON:
            keep_boxes.append(xyxy.cpu().tolist())
            keep_scores.append(float(conf.item()))
            keep_labels.append(YOLO_TO_COMMON[coco_id])

    if not keep_boxes:
        return _empty_pred()

    return {
        "boxes":  torch.tensor(keep_boxes,  dtype=torch.float32),
        "scores": torch.tensor(keep_scores, dtype=torch.float32),
        "labels": torch.tensor(keep_labels, dtype=torch.int64),
    }


# ── Faster R-CNN ─────────────────────────────────────────────────────────────

# def load_frcnn(device: str):
#     weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
#     model = fasterrcnn_resnet50_fpn(weights=weights)
#     model.eval().to(device)
#     return model, weights.transforms()


@torch.no_grad()
def predict_frcnn(
    model, preprocess, image: Image.Image, device: str, score_thr: float = 0.05
) -> dict:
    tensor = preprocess(image).to(device)
    output = model([tensor])[0]

    keep_boxes, keep_scores, keep_labels = [], [], []
    for box, score, label in zip(
        output["boxes"], output["scores"], output["labels"]
    ):
        coco_id = int(label.item())
        if float(score.item()) >= score_thr and coco_id in FRCNN_TO_COMMON:
            keep_boxes.append(box.cpu().tolist())
            keep_scores.append(float(score.item()))
            keep_labels.append(FRCNN_TO_COMMON[coco_id])

    if not keep_boxes:
        return _empty_pred()

    return {
        "boxes":  torch.tensor(keep_boxes,  dtype=torch.float32),
        "scores": torch.tensor(keep_scores, dtype=torch.float32),
        "labels": torch.tensor(keep_labels, dtype=torch.int64),
    }


# ── Основной цикл оценки ─────────────────────────────────────────────────────

def evaluate_dataset(
    dataset,
    detector_name: str,
    predict_fn,
    weather_system,
    limit: int = None,
) -> dict:
    metric_baseline = MeanAveragePrecision(box_format="xyxy")
    metric_system   = MeanAveragePrecision(box_format="xyxy")

    n = len(dataset) if limit is None else min(limit, len(dataset))

    for idx in range(n):
        image, target, path = dataset[idx]

        if target["boxes"].numel() == 0:
            continue

        # Baseline
        pred_baseline = predict_fn(image)
        metric_baseline.update([pred_baseline], [target])

        # С системой
        restored = weather_system.restore(image)
        pred_system = predict_fn(restored)
        metric_system.update([pred_system], [target])

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{n} изображений обработано")

    res_b = metric_baseline.compute()
    res_s = metric_system.compute()

    return {
        "detector":        detector_name,
        "n_images":        n,
        "baseline_mAP50":    float(res_b["map_50"]),
        "baseline_mAP50_95": float(res_b["map"]),
        "system_mAP50":      float(res_s["map_50"]),
        "system_mAP50_95":   float(res_s["map"]),
        "delta_mAP50":       float(res_s["map_50"]) - float(res_b["map_50"]),
        "delta_mAP50_95":    float(res_s["map"])    - float(res_b["map"]),
    }


def print_result(result: dict, dataset_name: str):
    d = result["detector"]
    n = result["n_images"]
    print(f"\n{'=' * 55}")
    print(f"  {d}  |  {dataset_name}  |  {n} изображений")
    print(f"{'=' * 55}")
    print(f"{'Режим':<28} {'mAP50':>8} {'mAP50-95':>10}")
    print(f"{'-' * 55}")
    print(f"{'Без системы (baseline)':<28}"
          f" {result['baseline_mAP50']:>8.4f}"
          f" {result['baseline_mAP50_95']:>10.4f}")
    print(f"{'С системой':<28}"
          f" {result['system_mAP50']:>8.4f}"
          f" {result['system_mAP50_95']:>10.4f}")
    print(f"{'Прирост':.<28}"
          f" {result['delta_mAP50']:>+8.4f}"
          f" {result['delta_mAP50_95']:>+10.4f}")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Устройство: {device}")

    # ── Система ───────────────────────────────────────────────────────────────
    system = WeatherRestorationSystem(device=device)

    # ── Детекторы ─────────────────────────────────────────────────────────────
    print("\nЗагрузка YOLOv8s...")
    yolo = load_yolo()

    print("Загрузка Faster R-CNN...")
    #frcnn_model, frcnn_preprocess = load_frcnn(device)

    def yolo_predict(image):
        return predict_yolo(yolo, image, device)

    # def frcnn_predict(image):
    #     return predict_frcnn(frcnn_model, frcnn_preprocess, image, device)

    # ── Датасеты ──────────────────────────────────────────────────────────────
    # Структура:
    #   data/DAWN/
    #       images/  ← fog_001.jpg, rain_002.jpg, snow_003.jpg, ...
    #       labels/  ← fog_001.txt, rain_002.txt, snow_003.txt, ...

    root_dawn = "./DAWN"

    # Вариант 1: Загружаем три раза с фильтром по типу погоды
    datasets_config = [
        ("DAWN (туман)",  load_dawn(root_dawn, weather_type="fogsmog")),
        ("DAWN (дождь)",  load_dawn(root_dawn, weather_type="rain")),
        ("DAWN (снег)",   load_dawn(root_dawn, weather_type="snow")),
    ]

    # Вариант 2: Если хочешь загрузить всё сразу (без разделения по типам)
    # datasets_config = [
    #     ("DAWN (все)",  load_dawn(root_dawn)),
    # ]

    # ── Прогон ────────────────────────────────────────────────────────────────
    # limit=None — весь набор; limit=50 — быстрая проверка
    limit = None

    for dataset_name, dataset in datasets_config:
        print(f"\n>>> {dataset_name}: {len(dataset)} изображений")

        for detector_name, predict_fn in [
            ("YOLOv8s",      yolo_predict),
            #("Faster R-CNN", frcnn_predict),
        ]:
            print(f"  Детектор: {detector_name}")
            result = evaluate_dataset(
                dataset, detector_name, predict_fn,
                system, limit=limit,
            )
            print_result(result, dataset_name)


if __name__ == "__main__":
    main()