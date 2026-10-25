"""Image -> embedding on the GPU (CPU fallback), with a choice of models.

Walkthrough frames and live screenshots go through exactly the same path:
320x180 RGB (the caller downsizes) -> squashed to the model's square input ->
normalised -> encoded. Squashing instead of centre-cropping keeps the whole
16:9 picture, which matters more than aspect ratio for "is this the same shot".

Every game remembers which model analysed its walkthrough (meta "model_key"),
so changing the default never mixes embeddings from two models.
"""
from __future__ import annotations

import threading

import numpy as np

from . import config as C

# key -> how to build it. "open_clip" models use their own image tower;
# "timm" DINOv2 models use [CLS token, mean of patch tokens].
MODELS = {
    "clip-b16": dict(kind="open_clip", name="ViT-B-16", pretrained="laion2b_s34b_b88k",
                     label="CLIP ViT-B/16 (LAION-2B)"),
    "clip-l14": dict(kind="open_clip", name="ViT-L-14", pretrained="datacomp_xl_s13b_b90k",
                     label="CLIP ViT-L/14 (DataComp-XL)"),
    "siglip2-b16": dict(kind="open_clip", name="ViT-B-16-SigLIP2", pretrained="webli",
                        label="SigLIP 2 ViT-B/16"),
    "dinov2-b14": dict(kind="timm", name="vit_base_patch14_dinov2.lvd142m",
                       label="DINOv2 ViT-B/14"),
    "dinov2-l14": dict(kind="timm", name="vit_large_patch14_dinov2.lvd142m",
                       label="DINOv2 ViT-L/14"),
}
# DINOv2 ViT-B/14 won the benchmark on real play (best match within 60 s for
# 71% of screenshots vs 58% for CLIP ViT-B/16; the right moment stands out
# far more: z 5.1 vs 3.75). Games analysed before it existed used CLIP.
DEFAULT_KEY = "dinov2-b14"
LEGACY_KEY = "clip-b16"

_instances: dict[str, "Embedder"] = {}
_instance_lock = threading.Lock()


def get(key: str | None = None) -> "Embedder":
    """One shared model per key for the indexer and the live tracker."""
    key = key or DEFAULT_KEY
    with _instance_lock:
        if key not in _instances:
            _instances[key] = Embedder(key)
        return _instances[key]


def loaded(key: str | None = None) -> bool:
    return (key or DEFAULT_KEY) in _instances


def release(key: str) -> None:
    """Drop a cached model and give its GPU memory back."""
    with _instance_lock:
        e = _instances.pop(key, None)
    if e is not None:
        e.close()


class Embedder:
    def __init__(self, key: str = DEFAULT_KEY):
        import torch

        if key not in MODELS:
            raise ValueError(f"unknown model {key!r}")
        spec = MODELS[key]
        self.key = key
        self.label = spec["label"]
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = self.device == "cuda"
        self.kind = spec["kind"]
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if self.kind == "open_clip":
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                spec["name"], pretrained=spec["pretrained"], cache_dir=str(C.MODELS_DIR))
            cfg = getattr(model.visual, "preprocess_cfg", None) or {}
            mean = cfg.get("mean") or getattr(model.visual, "image_mean", None) \
                or (0.48145466, 0.4578275, 0.40821073)
            std = cfg.get("std") or getattr(model.visual, "image_std", None) \
                or (0.26862954, 0.26130258, 0.27577711)
            size = cfg.get("size") or getattr(model.visual, "image_size", 224)
        else:
            import timm
            kw = dict(pretrained=True, num_classes=0, img_size=224)
            try:
                model = timm.create_model(spec["name"], cache_dir=str(C.MODELS_DIR), **kw)
            except TypeError:  # older timm without cache_dir
                model = timm.create_model(spec["name"], **kw)
            dc = timm.data.resolve_model_data_config(model)
            mean, std, size = dc["mean"], dc["std"], 224
        if isinstance(size, int):
            size = (size, size)
        model.eval().to(self.device)
        if self.half:
            model.half()
        self.model = model
        self.size = tuple(int(s) for s in size)[-2:]
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.lock = threading.Lock()
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        C.log(f"model {key} ({self.label}) on {self.gpu_name}")

    def _features(self, x):
        if self.kind == "open_clip":
            return self.model.encode_image(x)
        f = self.model.forward_features(x)                 # (B, prefix + patches, D)
        n_pre = getattr(self.model, "num_prefix_tokens", 1)
        return self.torch.cat([f[:, 0], f[:, n_pre:].mean(dim=1)], dim=-1)

    def embed(self, frames: np.ndarray) -> np.ndarray:
        """frames: (B, H, W, 3) uint8 RGB -> (B, D) float32, L2-normalised."""
        torch = self.torch
        F = torch.nn.functional
        if frames.ndim == 3:
            frames = frames[None]
        with self.lock, torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            x = F.interpolate(x, size=self.size, mode="bicubic", antialias=True,
                              align_corners=False).clamp_(0, 1)
            x = (x - self.mean) / self.std
            if self.half:
                x = x.half()
            f = self._features(x).float()
            f = F.normalize(f, dim=-1)
            return f.cpu().numpy()

    def close(self) -> None:
        try:
            del self.model
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
        except Exception:
            pass
