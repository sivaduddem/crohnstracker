import json
from pathlib import Path

from ml.data.label_map import is_target, map_label


def test_label_mapping():
    assert map_label("Ulcer") == "ulcer"
    assert map_label("Normal clean mucosa") == "normal"
    assert map_label("Polyp") == "other_unmapped"
    assert is_target("erosion")
    assert not is_target("other_unmapped")


def test_video_level_split_no_leakage(tmp_path: Path):
    manifest = {
        "cases": [
            {"id": "a1", "video_id": "v1", "gt_label": "ulcer", "path": "images/a1.jpg", "source": "t", "frame_idx": 0},
            {"id": "a2", "video_id": "v1", "gt_label": "ulcer", "path": "images/a2.jpg", "source": "t", "frame_idx": 1},
            {"id": "b1", "video_id": "v2", "gt_label": "normal", "path": "images/b1.jpg", "source": "t", "frame_idx": 0},
            {"id": "c1", "video_id": "v3", "gt_label": "erosion", "path": "images/c1.jpg", "source": "t", "frame_idx": 0},
            {"id": "d1", "video_id": "v4", "gt_label": "erythema", "path": "images/d1.jpg", "source": "t", "frame_idx": 0},
        ]
    }
    man_path = tmp_path / "manifest.json"
    man_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "splits.json"

    import ml.data.split_by_video as split_mod

    # Run split logic via subprocess-like call of main with patched argv
    import sys

    sys.argv = ["split_by_video.py", "--manifest", str(man_path), "--out", str(out), "--seed", "0"]
    split_mod.main()
    payload = json.loads(out.read_text(encoding="utf-8"))

    # Frames from same video must share split
    assert payload["case_split"]["a1"] == payload["case_split"]["a2"]

    # No overlapping case ids across splits
    train, val, test = set(payload["splits"]["train"]), set(payload["splits"]["val"]), set(payload["splits"]["test"])
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
