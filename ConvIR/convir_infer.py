"""
convir_infer.py — inference wrapper for ConvIR
Репозиторий: https://github.com/c-yn/ConvIR

Поддерживаемые задачи
---------------------
  "desnow"      → Image_desnowing/
  "dehaze_its"  → Dehazing/ITS/
  "dehaze_ots"  → Dehazing/OTS/

Пример использования
---------------------
    from convir_infer import ConvIRInference

    model = ConvIRInference(
        task="desnow",
        weights_path="./Image_desnowing/CSD.pkl",
        convir_root=".",     # путь к клонированному репозиторию
        version="base",      # small | base | large
        device="mps",        # mps | cuda | cpu
    )
    result = model("./snow.jpg")
    result.save("./clean.jpg")
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


# Соответствие задачи → подпапка в репозитории
_TASK_DIR = {
    "desnow":     "Image_desnowing",
    "dehaze_its": "Dehazing/ITS",
    "dehaze_ots": "Dehazing/OTS",
}

# Модель требует, чтобы стороны изображения были кратны этому числу
_PAD_FACTOR = 32


class ConvIRInference:
    """
    Обёртка вокруг ConvIR для инференса одного изображения.

    Параметры
    ---------
    task : "desnow" | "dehaze_its" | "dehaze_ots"
    weights_path : str
        Путь к файлу весов (.pkl / .pth).
    convir_root : str
        Корень клонированного репозитория ConvIR.
    version : "small" | "base" | "large"
        Размер модели — должен совпадать с тем, на котором обучены веса.
    device : str | None
        "mps", "cuda", "cpu" или None (авто-определение).
    """

    def __init__(
        self,
        task: Literal["desnow", "dehaze_its", "dehaze_ots"],
        weights_path: str,
        convir_root: str = ".",
        version: Literal["small", "base", "large"] = "base",
        device: str | None = None,
    ):
        if task not in _TASK_DIR:
            raise ValueError(f"task must be one of {list(_TASK_DIR)}, got '{task}'")

        # ── device ─────────────────────────────────────────────────────────
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # ── найти папку задачи ──────────────────────────────────────────────
        root = Path(convir_root).expanduser().resolve()
        task_dir = root / _TASK_DIR[task]
        if not task_dir.exists():
            raise FileNotFoundError(
                f"Папка задачи не найдена: {task_dir}\n"
                f"Убедись, что convir_root указывает на клонированный репозиторий."
            )

        # Добавляем папку задачи в sys.path, чтобы работал
        # относительный импорт внутри models/ConvIR.py:  from .layers import *
        task_dir_str = str(task_dir)
        if task_dir_str not in sys.path:
            sys.path.insert(0, task_dir_str)

        # ── импортировать build_net из models/ConvIR.py ─────────────────────
        net_path = task_dir / "models" / "ConvIR.py"
        if not net_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {net_path}")

        spec = importlib.util.spec_from_file_location("convir_model", str(net_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # ── собрать и загрузить модель ──────────────────────────────────────
        self.net = module.build_net(version)

        # Чекпоинт сохранён как словарь {'model': state_dict, ...}
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model"])

        self.net.to(self.device).eval()
        print(f"[ConvIR] Готово. Задача: {task}, версия: {version}, устройство: {self.device}")

    # ── публичный API ───────────────────────────────────────────────────────
    def __call__(self, img: Image) -> Image.Image:
        """
        Запустить восстановление изображения.

        Параметры
        ---------
        image_path : str  Путь к входному (деградированному) изображению.

        Возвращает
        ----------
        PIL.Image.Image  Восстановленное изображение (то же разрешение).
        """
        inp = TF.to_tensor(img).unsqueeze(0).to(self.device)  # 1×3×H×W  [0,1]

        h, w = inp.shape[2], inp.shape[3]

        # Паддинг до кратного _PAD_FACTOR — точно как в оригинальном eval.py
        H = ((h + _PAD_FACTOR) // _PAD_FACTOR) * _PAD_FACTOR
        W = ((w + _PAD_FACTOR) // _PAD_FACTOR) * _PAD_FACTOR
        padh = H - h if h % _PAD_FACTOR != 0 else 0
        padw = W - w if w % _PAD_FACTOR != 0 else 0
        inp_padded = F.pad(inp, (0, padw, 0, padh), mode="reflect")

        with torch.no_grad():
            # Модель возвращает список из 3 тензоров:
            #   outputs[0] → 1/4 разрешение
            #   outputs[1] → 1/2 разрешение
            #   outputs[2] → полное разрешение  ← берём этот
            outputs = self.net(inp_padded)
            pred = outputs[2]

        # Обрезаем обратно до исходного размера
        pred = pred[:, :, :h, :w]
        pred = torch.clamp(pred, 0, 1)

        # Поправка на квантование (из оригинального eval.py)
        pred += 0.5 / 255

        return TF.to_pil_image(pred.squeeze(0).cpu(), "RGB")


# ─────────────────────────────────────────────────────────────────────────────
# CLI:  python convir_infer.py --task desnow --weights ./CSD.pkl --input snow.jpg
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ConvIR — инференс одного изображения")
    parser.add_argument(
        "--task",
        required=True,
        choices=["desnow", "dehaze_its", "dehaze_ots"],
        help="Задача восстановления",
    )
    parser.add_argument("--weights",     required=True, help="Путь к файлу весов (.pkl/.pth)")
    parser.add_argument("--input",       required=True, help="Путь к входному изображению")
    parser.add_argument("--output",      default=None,  help="Путь для сохранения результата")
    parser.add_argument("--convir_root", default=".",   help="Корень репозитория ConvIR (default: .)")
    parser.add_argument(
        "--version",
        default="base",
        choices=["small", "base", "large"],
        help="Размер модели (default: base)",
    )
    parser.add_argument("--device", default=None, help="mps | cuda | cpu (авто если не указано)")

    args = parser.parse_args()

    output_path = args.output or str(
        Path(args.input).parent / (Path(args.input).stem + "_restored.png")
    )

    model = ConvIRInference(
        task=args.task,
        weights_path=args.weights,
        convir_root=args.convir_root,
        version=args.version,
        device=args.device,
    )

    print(f"[ConvIR] Обрабатываю {args.input} ...")
    result = model(args.input)
    result.save(output_path)
    print(f"[ConvIR] Сохранено → {output_path}")
