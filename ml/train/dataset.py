from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from crohnstracker_shared.constants import TARGET_LABELS


class LesionDataset(Dataset):
    def __init__(self, processed_dir: Path, case_ids: list[str], train: bool = False) -> None:
        self.root = processed_dir
        manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
        by_id = {c["id"]: c for c in manifest["cases"]}
        self.cases = [by_id[i] for i in case_ids if i in by_id]
        self.label_to_idx = {lab: i for i, lab in enumerate(TARGET_LABELS)}
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        c = self.cases[idx]
        img = Image.open(self.root / c["path"]).convert("RGB")
        y = self.label_to_idx.get(c["gt_label"], 0)
        return self.tf(img), y, c["id"]
