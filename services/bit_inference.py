from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIT_ROOT = PROJECT_ROOT / "third_party" / "bit"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_ckpt.pt"

# The original BiT implementation imports `models.*` and `misc.*`
# as top-level packages, so the vendored BIT directory must be on
# sys.path before those imports.
import sys

bit_root_str = str(BIT_ROOT)
if bit_root_str not in sys.path:
    sys.path.insert(0, bit_root_str)

from models.networks import define_G  # noqa: E402


class BiTInferenceAdapter:
    """Direct inference wrapper around the official pretrained BiT model."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"BiT checkpoint not found: {self.checkpoint_path}"
            )

        args = SimpleNamespace(
            n_class=2,
            net_G="base_transformer_pos_s4_dd8_dedim8",
            gpu_ids=[],
        )

        self.device = torch.device("cpu")

        self.model = define_G(
            args=args,
            init_type="normal",
            init_gain=0.02,
            gpu_ids=[],
        )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if "model_G_state_dict" not in checkpoint:
            raise ValueError(
                "Invalid BiT checkpoint: missing 'model_G_state_dict'."
            )

        self.model.load_state_dict(
            checkpoint["model_G_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _prepare_raster(raster: np.ndarray) -> torch.Tensor:
        if not isinstance(raster, np.ndarray):
            raise TypeError("Raster must be a NumPy array.")

        if raster.size == 0:
            raise ValueError("Raster is empty.")

        if raster.ndim != 3:
            raise ValueError(
                "Raster must have shape (bands, height, width)."
            )

        bands, height, width = raster.shape

        if bands < 1 or height < 1 or width < 1:
            raise ValueError("Invalid raster dimensions.")

        raster = raster.astype(np.float32, copy=True)

        finite_mask = np.isfinite(raster)

        if not finite_mask.any():
            raise ValueError("Raster contains no finite values.")

        replacement = float(np.median(raster[finite_mask]))
        raster[~finite_mask] = replacement

        # BiT expects 3-channel RGB input.
        if bands == 1:
            raster = np.repeat(raster, 3, axis=0)
        elif bands == 2:
            raster = np.concatenate([raster, raster[1:2]], axis=0)
        elif bands >= 3:
            raster = raster[:3]

        # Convert CHW -> HWC for PIL.
        image = np.transpose(raster, (1, 2, 0))

        # Normalize arbitrary raster values into the image-like range
        # expected by the original BiT preprocessing.
        image_min = float(np.min(image))
        image_max = float(np.max(image))

        if image_max - image_min < 1e-6:
            image = np.zeros_like(image)
        else:
            image = (
                (image - image_min)
                / (image_max - image_min)
                * 255.0
            )

        image = np.clip(image, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(image, mode="RGB")

        pil_image = TF.resize(
            pil_image,
            [256, 256],
            interpolation=3,
        )

        tensor = TF.to_tensor(pil_image)

        # Original BiT preprocessing:
        # (pixel / 255 - 0.5) / 0.5
        tensor = TF.normalize(
            tensor,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        return tensor.unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        raster_t1: np.ndarray,
        raster_t2: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        tensor_t1 = self._prepare_raster(raster_t1)
        tensor_t2 = self._prepare_raster(raster_t2)

        logits = self.model(
            tensor_t1.to(self.device),
            tensor_t2.to(self.device),
        )

        mask = (
            torch.argmax(logits, dim=1)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        change_percentage = float(
            np.mean(mask == 1) * 100.0
        )

        return mask, change_percentage