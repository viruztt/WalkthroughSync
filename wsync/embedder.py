"""Image -> embedding with an OpenCLIP model on the GPU (CPU fallback).

Walkthrough frames and live screenshots go through exactly the same path:
320x180 RGB (the caller downsizes) -> squashed to the model's square input ->
normalised -> encoded. Squashing instead of centre-cropping keeps the whole
16:9 picture, which matters more than aspect ratio for "is this the same shot".
"""
from __future__ import annotations

import threading

import numpy as np

from . import config as C

MODEL = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"

_instance = None
_instance_lock = threading.Lock()


def get() -> "Embedder":
    """One shared model for the indexer and the live tracker."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = Embedder()
        return _instance


def loaded() -> bool:
    return _instance is not None


class Embedder:
    def __init__(self):
        import torch
        import open_clip

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model, _, _ = open_clip.create_model_and_transforms(
            MODEL, pretrained=PRETRAINED, cache_dir=str(C.MODELS_DIR))
        model.eval().to(self.device)
        self.half = self.device == "cuda"
        if self.half:
            model.half()
        self.model = model

        cfg = getattr(model.visual, "preprocess_cfg", None) or {}
        mean = cfg.get("mean") or getattr(model.visual, "image_mean", None) \
            or (0.48145466, 0.4578275, 0.40821073)
        std = cfg.get("std") or getattr(model.visual, "image_std", None) \
            or (0.26862954, 0.26130258, 0.27577711)
        size = cfg.get("size") or getattr(model.visual, "image_size", 224)
        if isinstance(size, int):
            size = (size, size)
        self.size = tuple(int(s) for s in size)
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.lock = threading.Lock()
        self.gpu_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU"
        C.log(f"model {MODEL}/{PRETRAINED} on {self.gpu_name}")

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
            f = self.model.encode_image(x).float()
            f = F.normalize(f, dim=-1)
            return f.cpu().numpy()
