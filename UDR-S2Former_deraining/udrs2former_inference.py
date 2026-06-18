"""
udrs2former_inference.py
────────────────────────
Инференс UDR-S2Former (ICCV 2023).
Python 3.10+ / PyTorch 2.x / MPS или CPU.

Ключевые особенности модели, учтённые здесь:
  1. UDR_S2Former.py использует глобальную переменную `device` —
     инжектируем её в модуль до загрузки через importlib.
  2. forward() возвращает ([out_final, out0..out3], [var...]) —
     берём output[0][0] как финальный результат.
  3. img_size передаётся в __init__ и влияет на precomputed coords —
     должен совпадать с размером входного тайла/изображения.
  4. Паддинг до кратного 64 (patch_size модели) делается вручную.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ImageLike = Union[str, Path, np.ndarray, Image.Image]


# ─── утилиты ──────────────────────────────────────────────────────────────────

def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_transformer_class(arch_path: Path, device: torch.device):
    """
    Загружает класс Transformer из UDR_S2Former.py.
    Инжектирует `device` до exec_module — закрывает NameError в SparseSamplingAttention.
    Добавляет папку репозитория в sys.path для локальных импортов
    (base_net_snow.py, condconv.py и т.д.).
    """
    arch_path = arch_path.resolve()
    if not arch_path.exists():
        raise FileNotFoundError(f"Файл архитектуры не найден: {arch_path}")

    repo_dir = str(arch_path.parent)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    spec   = importlib.util.spec_from_file_location("UDR_S2Former", arch_path)
    module = importlib.util.module_from_spec(spec)
    module.device = device          # ← ключевой фикс
    spec.loader.exec_module(module)

    if not hasattr(module, "Transformer"):
        raise AttributeError(f"Класс 'Transformer' не найден в {arch_path}")
    return module.Transformer


def _to_tensor(img: ImageLike) -> torch.Tensor:
    if isinstance(img, (str, Path)):
        arr = cv2.imread(str(img), cv2.IMREAD_COLOR)
        if arr is None:
            raise FileNotFoundError(f"Не удалось открыть: {img}")
        img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    return (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)


def _pad64(x: torch.Tensor):
    """Паддинг до кратного 64. Возвращает тензор и оригинальный (H, W)."""
    _, _, H, W = x.shape
    ph = (64 - H % 64) % 64
    pw = (64 - W % 64) % 64
    return F.pad(x, (0, pw, 0, ph), mode="reflect"), (H, W)


# ─── tile-инференс ────────────────────────────────────────────────────────────

def _tile_infer(
    make_model,             # callable(img_size) → eval model на нужном device
    x: torch.Tensor,
    tile: int,
    overlap: int,
) -> torch.Tensor:
    """
    Tile-инференс.
    Модель создаётся через make_model(img_size=(tile, tile)) — это необходимо,
    потому что SparseSamplingAttention вычисляет base_coords в __init__
    под конкретный img_size.
    """
    B, C, H, W = x.shape
    assert tile % 64 == 0, "tile_size должен делиться на 64"

    stride   = tile - overlap
    h_starts = list(range(0, H - tile, stride)) + [H - tile]
    w_starts = list(range(0, W - tile, stride)) + [W - tile]

    tile_model = make_model((tile, tile))

    E = torch.zeros(B, C, H, W, dtype=x.dtype, device=x.device)
    N = torch.zeros_like(E)

    for hs in h_starts:
        for ws in w_starts:
            patch = x[..., hs:hs + tile, ws:ws + tile]
            with torch.no_grad():
                out_list, _ = tile_model(patch)
                out = out_list[0]
            E[..., hs:hs + tile, ws:ws + tile] += out
            N[..., hs:hs + tile, ws:ws + tile] += 1

    return E / N


# ─── основной класс ───────────────────────────────────────────────────────────

class UDRS2FormerInference:
    """
    Инференс UDR-S2Former.

    Параметры
    ---------
    weights_path : путь к .pth файлу весов
    arch_path    : путь к UDR_S2Former.py.
                   По умолчанию ищет в той же папке, что и этот файл.
    img_size     : (H, W) входного изображения.
                   При tile_size=None — изображение паддится до img_size,
                   если оно меньше, или обрабатывается как есть (с паддингом до 64).
                   При tile_size!=None — img_size игнорируется.
    tile_size    : размер тайла в пикселях, кратный 64 (например 320).
                   None — обработка целым изображением.
    tile_overlap : перекрытие тайлов (px). 0 работает нормально для этой модели.
    device       : 'cuda' / 'mps' / 'cpu' / None (автовыбор).

    Пример
    ------
    model = UDRS2FormerInference(
        weights_path = "pretrained/udrs2former_demo.pth",
        img_size     = (320, 320),
        tile_size    = 320,
        device       = "mps",
    )
    result = model("image_demo/input_images/rain.jpg")
    result.save("output/clean.png")
    """

    def __init__(
        self,
        weights_path:  str | Path,
        arch_path:     str | Path | None = None,
        img_size:      tuple[int, int]   = (320, 320),
        tile_size:     int | None        = None,
        tile_overlap:  int               = 0,
        device:        str | None        = None,
    ):
        self.device       = torch.device(device) if device else _auto_device()
        self.img_size     = img_size
        self.tile_size    = tile_size
        self.tile_overlap = tile_overlap

        if arch_path is None:
            arch_path = Path(__file__).parent / "UDR_S2Former.py"
        self._arch_path = Path(arch_path)

        Transformer = _load_transformer_class(self._arch_path, self.device)

        # factory для tile-режима: создаёт свежую модель под нужный img_size
        def _make_model(img_size):
            cls = _load_transformer_class(self._arch_path, self.device)
            m   = cls(img_size).to(self.device)
            self._apply_weights(m)
            m.eval()
            return m

        self._make_model = _make_model

        # основная модель для режима без тайлинга
        self.model = Transformer(img_size).to(self.device)
        self._load_weights(weights_path)
        self.model.eval()

        print(
            f"[UDR-S2Former] device={self.device} | img_size={img_size} | "
            f"tile={'{}px'.format(tile_size) if tile_size else 'off'}"
        )

    def _load_weights(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint не найден: {path}")
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict):
            state = state.get("params", state.get("state_dict", state))
        self._state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = self.model.load_state_dict(self._state, strict=True)
        if missing:
            print(f"  [warn] missing: {missing[:3]} ...")
        if unexpected:
            print(f"  [warn] unexpected: {unexpected[:3]} ...")

    def _apply_weights(self, model: torch.nn.Module) -> None:
        """Применяет сохранённые веса к произвольному экземпляру модели."""
        model.load_state_dict(self._state, strict=True)

    def _run(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(self.device)
        t, (H, W) = _pad64(t)

        with torch.no_grad():
            if self.tile_size:
                out = _tile_infer(
                    self._make_model, t,
                    self.tile_size, self.tile_overlap,
                )
            else:
                out_list, _ = self.model(t)
                out = out_list[0]

        return torch.clamp(out[:, :, :H, :W], 0, 1)

    # ─── публичный API ────────────────────────────────────────────────────────

    def __call__(self, img: ImageLike) -> Image.Image:
        """img (путь / numpy / PIL) → PIL.Image."""
        return Image.fromarray(_to_numpy(self._run(_to_tensor(img))))

    def restore_array(self, bgr: np.ndarray) -> np.ndarray:
        """OpenCV BGR uint8 → BGR uint8."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(_to_numpy(self._run(_to_tensor(rgb))), cv2.COLOR_RGB2BGR)

    def restore_file(self, src: str | Path, dst: str | Path, quality: int = 95) -> Path:
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        self(src).save(dst, quality=quality)
        print(f"  saved → {dst}")
        return dst

    def restore_batch(
        self,
        in_dir:  str | Path,
        out_dir: str | Path,
        exts:    tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> list[Path]:
        in_dir  = Path(in_dir)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in sorted(in_dir.iterdir()) if f.suffix.lower() in exts]
        saved = []
        for i, f in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] {f.name}")
            saved.append(self.restore_file(f, out_dir / f.name))
        return saved
