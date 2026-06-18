from pathlib import Path
from udrs2former_inference import UDRS2FormerInference

HERE = Path(__file__).parent

model = UDRS2FormerInference(
    weights_path = HERE / "pretrained/udrs2former_raindrop_syn.pth",
    arch_path    = HERE / "UDR_S2Former.py",  # можно не указывать — найдёт сам
    img_size     = (320, 320),  # из demo.yaml: img_size_h / img_size_w
    tile_size    = 320,         # равен img_size — тайлинг по размеру патча
    tile_overlap = 0,
    device       = "mps",       # или "cpu"
)

# result = model(HERE / "./rain.jpg")
# result.save(HERE / "./clean.png")

import math, time, platform
from pathlib import Path
import numpy as np, cv2, torch
import lpips as lpips_lib


MODEL_NAME = "UDRS2Former"
DEVICE = "mps"

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# INPUT_DIR = "./WeatherBench/test/haze/input"
# TARGET_DIR = "./WeatherBench/test/haze/target"
INPUT_DIR = BASE_DIR / "WeatherBench" / "rain" / "test" / "input"
TARGET_DIR = BASE_DIR / "WeatherBench" / "rain" / "test" / "target"
CROP_BORDER = 4  # как в официальном evaluation.py WeatherBench
# ----------------------------------------------------------------------------


def _bgr2y(img):
    """BGR float [0,1] → Y float [0,255], как в официальном скрипте WeatherBench."""
    return np.dot(img * 255., [24.966, 128.553, 65.481]) / 255.0 + 16.0


def _psnr(pred, gt):
    """pred, gt — Y-канал float."""
    mse = np.mean((pred - gt) ** 2)
    return float('inf') if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))


def _ssim(img1, img2):
    """img1, img2 — Y-канал uint8 [0,255]."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    i1, i2 = img1.astype(np.float64), img2.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.T)
    mu1 = cv2.filter2D(i1, -1, w)[5:-5, 5:-5]
    mu2 = cv2.filter2D(i2, -1, w)[5:-5, 5:-5]
    s1 = cv2.filter2D(i1 ** 2, -1, w)[5:-5, 5:-5] - mu1 ** 2
    s2 = cv2.filter2D(i2 ** 2, -1, w)[5:-5, 5:-5] - mu2 ** 2
    s12 = cv2.filter2D(i1 * i2, -1, w)[5:-5, 5:-5] - mu1 * mu2
    return float(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean() \
        if False else float(
        (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean())


_lpips_fn = lpips_lib.LPIPS(net="alex").to(DEVICE);
_lpips_fn.eval()


def _sync():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.synchronize()


input_paths = sorted(p for p in Path(INPUT_DIR).iterdir()
                     if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

# прогрев
print(f"Прогрев ({min(3, len(input_paths))} изображения)...")
for p in input_paths[:3]:
    model(str(p))

psnr_vals, ssim_vals, lpips_vals, times = [], [], [], []

for inp_path in input_paths:
    tgt_path = Path(TARGET_DIR) / inp_path.name
    if not tgt_path.exists():
        print(f"  [skip] {tgt_path.name}")
        continue

    gt_bgr = cv2.imread(str(tgt_path)).astype(np.float64) / 255.

    _sync()
    t0 = time.perf_counter()
    pred_pil = model(str(inp_path))
    _sync()
    times.append((time.perf_counter() - t0) * 1000)

    # PIL → BGR float [0,1]
    pred_bgr = cv2.cvtColor(np.array(pred_pil.convert("RGB")), cv2.COLOR_RGB2BGR).astype(np.float64) / 255.
    if pred_bgr.shape != gt_bgr.shape:
        pred_bgr = cv2.resize(pred_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]))

    # Y-канал для PSNR/SSIM (как в WeatherBench evaluation.py)
    pred_y = _bgr2y(pred_bgr)
    gt_y = _bgr2y(gt_bgr)
    c = CROP_BORDER
    pred_yc, gt_yc = pred_y[c:-c, c:-c], gt_y[c:-c, c:-c]

    psnr_vals.append(_psnr(pred_yc, gt_yc))
    ssim_vals.append(_ssim(pred_yc, gt_yc))


    # LPIPS — RGB tensor [-1,1]
    def _to_lpips(bgr):
        rgb = cv2.cvtColor((bgr * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 127.5 - 1.


    with torch.no_grad():
        lpips_vals.append(float(_lpips_fn(_to_lpips(pred_bgr), _to_lpips(gt_bgr)).item()))

print(f"\n{'─' * 45}")
print(f"Модель      : {MODEL_NAME}")
print(f"Устройство  : {DEVICE}  |  {platform.processor()}")
print(f"Изображений : {len(psnr_vals)}")
print(f"PSNR  (Y)   : {np.mean(psnr_vals):.4f}  (±{np.std(psnr_vals):.4f})")
print(f"SSIM  (Y)   : {np.mean(ssim_vals):.4f}  (±{np.std(ssim_vals):.4f})")
print(f"LPIPS       : {np.mean(lpips_vals):.4f}  (±{np.std(lpips_vals):.4f})")
print(f"Инференс    : {np.mean(times):.1f} мс/изобр  (±{np.std(times):.1f})")
print(f"{'─' * 45}")