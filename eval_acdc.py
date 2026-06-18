"""
run_eval.py — оценка влияния системы восстановления на детекцию объектов.

Датасет:    ACDC (val-сплит, fog / rain / snow)
Детекторы:  YOLO26s, Faster R-CNN ResNet50
Метрика:    mAP50, mAP50-95

pip install ultralytics torchvision torchmetrics pillow tqdm
"""

import ssl
import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
)
from PIL import Image

from weather_datasets_acdc import load_acdc, COMMON_ID, COMMON_CLASSES
from main import WeatherRestorationSystem

# SSL-фикс для macOS
ssl._create_default_https_context = ssl._create_unverified_context

# YOLO26/YOLOv8: COCO с нуля (0=person, 1=bicycle, 2=car, ...)
YOLO_TO_COMMON = {
    0: COMMON_ID["person"],
    1: COMMON_ID["bicycle"],
    2: COMMON_ID["car"],
    3: COMMON_ID["motorbike"],
    5: COMMON_ID["bus"],
}

# FasterRCNN: COCO с единицы (0=фон)
FRCNN_TO_COMMON = {
    1: COMMON_ID["person"],
    2: COMMON_ID["bicycle"],
    3: COMMON_ID["car"],
    4: COMMON_ID["motorbike"],
    6: COMMON_ID["bus"],
}


def empty_pred():
    return {
        "boxes":  torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,),   dtype=torch.float32),
        "labels": torch.zeros((0,),   dtype=torch.int64),
    }


# ── Модели ────────────────────────────────────────────────────────────────────

yolo = YOLO("yolo26s.pt")

frcnn_weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
frcnn         = fasterrcnn_resnet50_fpn(weights=frcnn_weights).eval()
frcnn_prep    = frcnn_weights.transforms()


def detect_yolo(image: Image.Image, device: str) -> dict:
    res = yolo.predict(image, device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return empty_pred()
    boxes, scores, labels = [], [], []
    for xyxy, conf, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
        cid = int(cls)
        if cid in YOLO_TO_COMMON:
            boxes.append(xyxy.cpu().tolist())
            scores.append(float(conf))
            labels.append(YOLO_TO_COMMON[cid])
    if not boxes:
        return empty_pred()
    return {
        "boxes":  torch.tensor(boxes,  dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


@torch.no_grad()
def detect_frcnn(image: Image.Image, device: str) -> dict:
    out = frcnn.to(device)([frcnn_prep(image).to(device)])[0]
    boxes, scores, labels = [], [], []
    for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
        cid = int(label)
        if float(score) >= 0.05 and cid in FRCNN_TO_COMMON:
            boxes.append(box.cpu().tolist())
            scores.append(float(score))
            labels.append(FRCNN_TO_COMMON[cid])
    if not boxes:
        return empty_pred()
    return {
        "boxes":  torch.tensor(boxes,  dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


# ── Оценка ────────────────────────────────────────────────────────────────────

def evaluate(dataset, detect_fn, system, device, desc, limit=None):
    m_base   = MeanAveragePrecision(box_format="xyxy")
    m_system = MeanAveragePrecision(box_format="xyxy")

    n = min(limit, len(dataset)) if limit else len(dataset)

    for i in tqdm(range(n), desc=desc, unit="img"):
        image, target, _ = dataset[i]
        if target["boxes"].numel() == 0:
            continue
        m_base.update([detect_fn(image,                   device)], [target])
        m_system.update([detect_fn(system.restore(image), device)], [target])

    b = m_base.compute()
    s = m_system.compute()
    return {
        "mAP50_base":    float(b["map_50"]),
        "mAP50_system":  float(s["map_50"]),
        "delta_mAP50":   float(s["map_50"]) - float(b["map_50"]),
        "mAP9595_base":  float(b["map"]),
        "mAP9595_system":float(s["map"]),
        "delta_mAP9595": float(s["map"])    - float(b["map"]),
    }


def print_result(name, detector, r):
    W = 58
    print(f"\n{'='*W}")
    print(f"  {detector}  |  {name}")
    print(f"{'='*W}")
    print(f"{'Режим':<26} {'mAP50':>8} {'mAP50-95':>10}")
    print(f"{'-'*W}")
    print(f"{'Без системы':<26} {r['mAP50_base']:>8.4f} {r['mAP9595_base']:>10.4f}")
    print(f"{'С системой':<26} {r['mAP50_system']:>8.4f} {r['mAP9595_system']:>10.4f}")
    s50   = "+" if r["delta_mAP50"]   >= 0 else ""
    s9595 = "+" if r["delta_mAP9595"] >= 0 else ""
    print(f"{'Прирост':.<26} {s50}{r['delta_mAP50']:>7.4f} {s9595}{r['delta_mAP9595']:>9.4f}")


def print_summary(all_results):
    W = 72
    print(f"\n{'#'*W}")
    print("  ИТОГО")
    print(f"{'#'*W}")
    print(f"{'Датасет':<16} {'Детектор':<14}"
          f" {'baseMap50':>7} {'sys50':>7} {'Δ50':>6}"
          f" {'baseMap50-95':>7} {'sys95':>7} {'Δ95':>6}")
    print("-" * W)
    for ds_name, det_name, r in all_results:
        s50   = "+" if r["delta_mAP50"]   >= 0 else ""
        s9595 = "+" if r["delta_mAP9595"] >= 0 else ""
        print(f"{ds_name:<16} {det_name:<14}"
              f" {r['mAP50_base']:>7.4f} {r['mAP50_system']:>7.4f} {s50}{r['delta_mAP50']:>5.4f}"
              f" {r['mAP9595_base']:>7.4f} {r['mAP9595_system']:>7.4f} {s9595}{r['delta_mAP9595']:>5.4f}")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    system = WeatherRestorationSystem(device=device)

    ROOT = "./ACDC"

    datasets = [
        ("ACDC туман",  load_acdc(ROOT, "fog")),
        ("ACDC дождь",  load_acdc(ROOT, "rain")),
        ("ACDC снег",   load_acdc(ROOT, "snow")),
    ]

    detectors = [
        ("YOLO26s",      detect_yolo),
        ("Faster R-CNN", detect_frcnn),
    ]

    # limit=None — полный прогон, limit=50 — быстрая проверка
    limit = None

    all_results = []
    for ds_name, dataset in datasets:
        for det_name, detect_fn in detectors:
            r = evaluate(
                dataset, detect_fn, system, device,
                desc=f"{det_name} | {ds_name}",
                limit=limit,
            )
            print_result(ds_name, det_name, r)
            all_results.append((ds_name, det_name, r))

    print_summary(all_results)


if __name__ == "__main__":
    main()