"""Real-package RF-DETR inference smoke coverage."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def test_rfdetr_seg_nano_constructs_and_predicts_deterministic_frame() -> None:
    """Exercise the public RF-DETR package without replacing its model or prediction boundary."""
    from detection.rfdetr_segmenter import RFDETRSegmenter
    from rfdetr import RFDETRSegNano

    height, width = 64, 96
    x_coords = np.broadcast_to(np.arange(width, dtype=np.uint8), (height, width))
    y_coords = np.broadcast_to(
        np.arange(height, dtype=np.uint8)[:, None], (height, width)
    )
    bgr_frame = np.stack(
        (
            x_coords,
            y_coords,
            (x_coords.astype(np.uint16) + y_coords.astype(np.uint16)).astype(np.uint8),
        ),
        axis=-1,
    )
    segmenter = RFDETRSegmenter(
        config={
            "rfdetr": {
                "model_variant": "nano",
                "pretrain_weights": None,
                "threshold": 1.0,
                "classes": None,
                "mask_erosion_px": 0,
                "mask_erosion_iters": 1,
            }
        },
        device="cpu",
    )

    segmenter.load_model()
    detections = segmenter.segment(bgr_frame, frame_id=12, timestamp=0.5)

    assert isinstance(segmenter.model, RFDETRSegNano)
    assert segmenter.class_id_to_name
    assert isinstance(detections, list)
    assert "predict" in segmenter.last_timings_seconds
    for detection in detections:
        assert detection.mask.shape == (height, width)
        assert detection.mask.dtype == bool
