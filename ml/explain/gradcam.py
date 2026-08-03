#!/usr/bin/env python3
"""Grad-CAM for EfficientNet-B0."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ml.train.model import EfficientNetEncoder


class GradCAM:
    def __init__(self, model: EfficientNetEncoder, target_layer_name: str = "blocks.6") -> None:
        self.model = model
        self.model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        layer = dict(model.backbone.named_modules())[target_layer_name]
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _forward_hook(self, _module, _inp, out):
        self.activations = out.detach()

    def _backward_hook(self, _module, _gin, gout):
        self.gradients = gout[0].detach()

    @torch.enable_grad()
    def generate(self, image: Image.Image, class_idx: int | None = None) -> np.ndarray:
        x = self.tf(image.convert("RGB")).unsqueeze(0)
        x.requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        assert self.gradients is not None and self.activations is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_cam(image: Image.Image, cam: np.ndarray, out_path: Path) -> None:
    img = image.convert("RGB").resize((224, 224))
    base = np.asarray(img).astype(np.float32)
    heat = np.zeros_like(base)
    heat[..., 0] = cam * 255
    heat[..., 1] = (1 - cam) * 80
    blended = (0.55 * base + 0.45 * heat).clip(0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blended).save(out_path)


def load_model(ckpt_path: Path, device: torch.device) -> EfficientNetEncoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    labels = ckpt.get("labels", ["normal", "erythema", "erosion", "ulcer"])
    model = EfficientNetEncoder(num_classes=len(labels))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
