"""
run_eval.py
===========
Оценка влияния системы восстановления изображений на детекцию объектов.

Датасет: ACDC (Kaggle: khalilusmanuk/acdc-clean-dataset-for-training)
         Тестовая выборка, три типа погоды: fog / rain / snow

Детекторы: YOLOv8s, Faster R-CNN ResNet50 FPN
Метрика:   mAP50, mAP50-95 (torchmetrics)

Зависимости:
    pip install ultralytics torchvision torchmetrics pillow tqdm
"""

import ssl
import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from PIL import Image

from weather_datasets_acdc import load_acdc, COMMON_ID, COMMON_CLASSES
from main import WeatherRestorationSystem

# ── SSL-фикс для macOS ────────────────────────────────────────────────────────
ssl._create_default_https_context = ssl._create_unverified_context

# ── Маппинг COCO-классов детекторов в COMMON_ID ───────────────────────────────

# YOLOv8: COCO с нуля
YOLO_COCO = {"person": 0, "bicycle": 1, "car": 2, "motorbike": 3, "bus": 5}
# FasterRCNN: COCO с единицы (0 = фон)
FRCNN_COCO = {"person": 1, "bicycle": 2, "car": 3, "motorbike": 4, "bus": 6}

YOLO_TO_COMMON  = {YOLO_COCO[n]:  COMMON_ID[n] for n in COMMON_CLASSES}
FRCNN_TO_COMMON = {FRCNN_COCO[n]: COMMON_ID[n] for n in COMMON_CLASSES}


def _empty_pred() -> dict:
    return {
        "boxes":  torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,),   dtype=torch.float32),
        "labels": torch.zeros((0,),   dtype=torch.int64),
    }


def load_yolo(model_name: str = "yolo26s.pt") -> YOLO:
    """Загружает YOLO26s (веса скачиваются автоматически)."""
    return YOLO(model_name)


def predict_yolo(model: YOLO, image: Image.Image, device: str) -> dict:
    results = model.predict(image, device=device, verbose=False)[0]
    bt = results.boxes
    if bt is None or len(bt) == 0:
        return _empty_pred()

    keep_boxes, keep_scores, keep_labels = [], [], []
    for xyxy, conf, cls in zip(bt.xyxy, bt.conf, bt.cls):
        cid = int(cls.item())
        if cid in YOLO_TO_COMMON:
            keep_boxes.append(xyxy.cpu().tolist())
            keep_scores.append(float(conf.item()))
            keep_labels.append(YOLO_TO_COMMON[cid])

    if not keep_boxes:
        return _empty_pred()
    return {
        "boxes":  torch.tensor(keep_boxes,  dtype=torch.float32),
        "scores": torch.tensor(keep_scores, dtype=torch.float32),
        "labels": torch.tensor(keep_labels, dtype=torch.int64),
    }


# ── Faster R-CNN ─────────────────────────────────────────────────────────────

def load_frcnn(device: str):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval().to(device)
    return model, weights.transforms()


@torch.no_grad()
def predict_frcnn(model, preprocess, image: Image.Image,
                  device: str, score_thr: float = 0.05) -> dict:
    tensor = preprocess(image).to(device)
    output = model([tensor])[0]

    keep_boxes, keep_scores, keep_labels = [], [], []
    for box, score, label in zip(
        output["boxes"], output["scores"], output["labels"]
    ):
        cid = int(label.item())
        if float(score.item()) >= score_thr and cid in FRCNN_TO_COMMON:
            keep_boxes.append(box.cpu().tolist())
            keep_scores.append(float(score.item()))
            keep_labels.append(FRCNN_TO_COMMON[cid])

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
    dataset_name: str,
    limit: int | None = None,
) -> dict:
    """
    Прогоняет детектор в двух режимах (baseline и с системой).

    Параметры
    ----------
    dataset       : датасет с методом __len__ / __getitem__
    detector_name : строка для вывода
    predict_fn    : callable(PIL.Image) -> dict{boxes, scores, labels}
    weather_system: объект с методом restore(PIL.Image) -> PIL.Image
    dataset_name  : строка для вывода и tqdm
    limit         : максимальное число изображений (None = все)
    """
    metric_baseline = MeanAveragePrecision(box_format="xyxy")
    metric_system   = MeanAveragePrecision(box_format="xyxy")

    n = len(dataset) if limit is None else min(limit, len(dataset))
    skipped = 0

    bar = tqdm(
        range(n),
        desc=f"  {detector_name:<14} {dataset_name}",
        unit="img",
        dynamic_ncols=True,
    )

    for idx in bar:
        image, target, _ = dataset[idx]

        # Пропускаем изображения без разметки
        if target["boxes"].numel() == 0:
            skipped += 1
            bar.set_postfix(skipped=skipped)
            continue

        # — Baseline —
        pred_b = predict_fn(image)
        metric_baseline.update([pred_b], [target])

        # — С системой —
        restored = weather_system.restore(image)
        pred_s = predict_fn(restored)
        metric_system.update([pred_s], [target])

    bar.close()

    res_b = metric_baseline.compute()
    res_s = metric_system.compute()

    return {
        "detector":          detector_name,
        "dataset":           dataset_name,
        "n_total":           n,
        "n_skipped":         skipped,
        "baseline_mAP50":    float(res_b["map_50"]),
        "baseline_mAP50_95": float(res_b["map"]),
        "system_mAP50":      float(res_s["map_50"]),
        "system_mAP50_95":   float(res_s["map"]),
        "delta_mAP50":       float(res_s["map_50"]) - float(res_b["map_50"]),
        "delta_mAP50_95":    float(res_s["map"])    - float(res_b["map"]),
    }


def print_result(r: dict):
    n_used = r["n_total"] - r["n_skipped"]
    print(f"\n{'=' * 58}")
    print(f"  {r['detector']}  |  {r['dataset']}  |  "
          f"{n_used} изображений (пропущено: {r['n_skipped']})")
    print(f"{'=' * 58}")
    print(f"{'Режим':<30} {'mAP50':>8} {'mAP50-95':>10}")
    print(f"{'-' * 58}")
    print(f"{'Без системы (baseline)':<30}"
          f" {r['baseline_mAP50']:>8.4f}"
          f" {r['baseline_mAP50_95']:>10.4f}")
    print(f"{'С системой':<30}"
          f" {r['system_mAP50']:>8.4f}"
          f" {r['system_mAP50_95']:>10.4f}")
    delta_sign = "+" if r["delta_mAP50"] >= 0 else ""
    print(f"{'Прирост':.<30}"
          f" {delta_sign}{r['delta_mAP50']:>7.4f}"
          f" {delta_sign}{r['delta_mAP50_95']:>9.4f}")


def print_summary(results: list[dict]):
    """Сводная таблица всех результатов."""
    print(f"\n{'#' * 58}")
    print("  СВОДНАЯ ТАБЛИЦА")
    print(f"{'#' * 58}")
    header = f"{'Датасет':<18} {'Детектор':<14} {'mAP50 base':>10} {'mAP50 sys':>10} {'Δ mAP50':>9}"
    print(header)
    print("-" * 58)
    for r in results:
        delta_str = f"{r['delta_mAP50']:+.4f}"
        print(f"{r['dataset']:<18} {r['detector']:<14}"
              f" {r['baseline_mAP50']:>10.4f}"
              f" {r['system_mAP50']:>10.4f}"
              f" {delta_str:>9}")


# ── Точка входа ───────────────────────────────────────────────────────────────

def main():
    # ── Устройство ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Устройство: {device}")

    # ── Система восстановления ────────────────────────────────────────────────
    print("Инициализация системы...")
    system = WeatherRestorationSystem(device=device)

    # ── Детекторы ─────────────────────────────────────────────────────────────
    print("\nЗагрузка YOLO26s...")
    yolo = load_yolo("yolo26s.pt")

    print("Загрузка Faster R-CNN...")
    frcnn_model, frcnn_preprocess = load_frcnn(device)

    def yolo_predict(img): return predict_yolo(yolo, img, device)
    def frcnn_predict(img): return predict_frcnn(frcnn_model, frcnn_preprocess, img, device)

    # ── Датасеты ──────────────────────────────────────────────────────────────
    # Укажи путь к распакованному датасету.
    # После kaggle datasets download -d khalilusmanuk/acdc-clean-dataset-for-training
    # распакуй и укажи корневую папку.
    ROOT = "./ACDC"

    print(f"\nЗагрузка датасетов из {ROOT}...")
    datasets_config = [
        ("ACDC (туман)",  load_acdc(ROOT, weather="fog",  split="test")),
        ("ACDC (дождь)",  load_acdc(ROOT, weather="rain", split="test")),
        ("ACDC (снег)",   load_acdc(ROOT, weather="snow", split="test")),
    ]

    # ── Прогон ────────────────────────────────────────────────────────────────
    # limit=None — весь test-сплит (рекомендуется для ВКР)
    # limit=50   — быстрая проверка что всё работает
    limit = None

    all_results = []

    for dataset_name, dataset in datasets_config:
        print(f"\n>>> {dataset_name}: {len(dataset)} изображений в тест-сплите")

        for det_name, predict_fn in [
            ("YOLOv11S",      yolo_predict),
            ("Faster R-CNN", frcnn_predict),
        ]:
            result = evaluate_dataset(
                dataset, det_name, predict_fn,
                system, dataset_name, limit=limit,
            )
            print_result(result)
            all_results.append(result)

    print_summary(all_results)


if __name__ == "__main__":
    main()