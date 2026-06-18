"""
DehazeFormer inference wrapper.
Repo: https://github.com/IDKiro/DehazeFormer

Usage:
    from dehazeformer_infer import DehazeFormerInference

    model = DehazeFormerInference(
        variant="b",
        weights_path="./save_models/indoor/dehazeformer-b.pth",
        dehazeformer_root="./DehazeFormer",
        device="mps",
    )
    result = model("./hazy.jpg")
    result.save("./clear.jpg")
"""

import importlib
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Available model variants (maps friendly name → factory function name)
# ---------------------------------------------------------------------------
VARIANTS = {
    "t":  "dehazeformer_t",
    "s":  "dehazeformer_s",
    "b":  "dehazeformer_b",
    "w":  "dehazeformer_w",
    "d":  "dehazeformer_d",
    "m":  "dehazeformer_m",
    "l":  "dehazeformer_l",
    "dehazeformer-t": "dehazeformer_t",
    "dehazeformer-s": "dehazeformer_s",
    "dehazeformer-b": "dehazeformer_b",
    "dehazeformer-w": "dehazeformer_w",
    "dehazeformer-d": "dehazeformer_d",
    "dehazeformer-m": "dehazeformer_m",
    "dehazeformer-l": "dehazeformer_l",
}

# DehazeFormer (Swin-based) processes images in windows of size 8.
# With 3 downsampling stages, the minimum safe alignment is 8 * 2^2 = 32.
_WINDOW_ALIGN = 32


def _load_state_dict(weights_path: str) -> OrderedDict:
    """Load checkpoint; strips DataParallel 'module.' prefix when present."""
    ckpt = torch.load(weights_path, map_location="cpu")
    raw = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

    new_sd = OrderedDict()
    for k, v in raw.items():
        name = k[7:] if k.startswith("module.") else k
        new_sd[name] = v
    return new_sd


def _load_model_module(dehazeformer_root: str):
    root = str(Path(dehazeformer_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("models")


def _pad_to_align(tensor: torch.Tensor, align: int = _WINDOW_ALIGN):
    """Pad [1, C, H, W] so H and W are divisible by `align`.

    Uses reflect padding so border regions are plausible inputs for the model
    rather than zeros (which would create visible seams).

    Returns (padded_tensor, orig_H, orig_W).
    """
    _, _, H, W = tensor.shape
    pad_h = (align - H % align) % align
    pad_w = (align - W % align) % align
    # F.pad order: (left, right, top, bottom)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, H, W


def _unpad(tensor: torch.Tensor, orig_H: int, orig_W: int) -> torch.Tensor:
    return tensor[:, :, :orig_H, :orig_W]


class DehazeFormerInference:
    """Thin inference wrapper around DehazeFormer.

    Parameters
    ----------
    variant : str
        Model size: one of 't', 's', 'b', 'w', 'd', 'm', 'l'
        (or full name, e.g. 'dehazeformer-b').
    weights_path : str
        Path to the pretrained .pth checkpoint.
    dehazeformer_root : str
        Path to the cloned DehazeFormer repo root
        (directory containing the `models/` folder).
    device : str
        PyTorch device string: 'cpu', 'cuda', 'mps', etc.
    tile : int | None
        If set, process image in overlapping tiles of this size (pixels).
        Must be a multiple of 32.  None = process full image.
    tile_overlap : int
        Overlap between tiles (pixels); only used when tile is not None.
    """

    def __init__(
        self,
        variant: str = "b",
        weights_path: str = "./save_models/indoor/dehazeformer-b.pth",
        dehazeformer_root: str = "./DehazeFormer",
        device: str = "cpu",
        tile: int | None = None,
        tile_overlap: int = 32,
    ):
        self.device = torch.device(device)
        self.tile = tile
        self.tile_overlap = tile_overlap

        if tile is not None and tile % _WINDOW_ALIGN != 0:
            raise ValueError(
                f"tile={tile} must be a multiple of {_WINDOW_ALIGN} "
                "(DehazeFormer Swin window-alignment requirement)."
            )

        fn_name = VARIANTS.get(variant)
        if fn_name is None:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from: {list(VARIANTS.keys())}"
            )

        models_mod = _load_model_module(dehazeformer_root)
        self.network = getattr(models_mod, fn_name)()

        state_dict = _load_state_dict(weights_path)
        missing, unexpected = self.network.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[DehazeFormerInference] Warning — missing keys:\n  {missing}")
        if unexpected:
            print(f"[DehazeFormerInference] Warning — unexpected keys:\n  {unexpected}")

        self.network.to(self.device)
        self.network.eval()

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """PIL RGB → float32 tensor in [-1, 1], shape [1, 3, H, W]."""
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5                                       # [-1, 1]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)    # [1,3,H,W]

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Model output [-1, 1] tensor → PIL RGB image."""
        out = tensor.clamp_(-1.0, 1.0) * 0.5 + 0.5                   # [0, 1]
        arr = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _forward_full(self, inp: torch.Tensor) -> torch.Tensor:
        """Forward pass with automatic padding/unpadding.

        Swin Transformer requires H and W to be multiples of window_size.
        Without padding, attention is computed over misaligned windows, which
        produces rectangular block artifacts and bright halos around objects.
        """
        padded, orig_H, orig_W = _pad_to_align(inp)
        padded = padded.to(self.device)
        with torch.no_grad():
            out_padded = self.network(padded)
        return _unpad(out_padded.cpu(), orig_H, orig_W)

    def _forward_tiled(self, inp: torch.Tensor) -> torch.Tensor:
        """Tiled forward pass for memory-constrained devices."""
        _, _, H, W = inp.shape
        tile = self.tile
        stride = tile - self.tile_overlap

        output = torch.zeros_like(inp)
        count  = torch.zeros_like(inp)

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(y + tile, H)
                x_end = min(x + tile, W)
                y0 = max(y_end - tile, 0)
                x0 = max(x_end - tile, 0)

                patch = inp[:, :, y0:y_end, x0:x_end].to(self.device)
                with torch.no_grad():
                    out_patch = self.network(patch).cpu()

                output[:, :, y0:y_end, x0:x_end] += out_patch
                count[:, :, y0:y_end, x0:x_end]  += 1

        return output / count

    # ---------------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------------- #

    def __call__(self, image_input) -> Image.Image:
        """Run dehazing inference.

        Parameters
        ----------
        image_input : str | Path | PIL.Image.Image | torch.Tensor
            File path, PIL image, or float32 tensor [1, 3, H, W] in [-1, 1].

        Returns
        -------
        PIL.Image.Image  —  dehazed RGB image.
        """
        if isinstance(image_input, (str, Path)):
            inp = self._pil_to_tensor(Image.open(image_input))
        elif isinstance(image_input, Image.Image):
            inp = self._pil_to_tensor(image_input)
        elif isinstance(image_input, torch.Tensor):
            inp = image_input.unsqueeze(0).float() if image_input.dim() == 3 else image_input.float()
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        out = self._forward_tiled(inp) if self.tile else self._forward_full(inp)
        return self._tensor_to_pil(out)
