"""Unified orchestrator running all 4 non-ML watermark detection layers."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image

from .detectors import (
    analyze_dsp_carriers,
    detect_c2pa,
    detect_container_signatures,
    detect_pixel_watermarks,
)


def analyze_image_bytes(
    raw_bytes: bytes, filename: str = "image", include_visuals: bool = True
) -> dict[str, Any]:
    """Run all 4 deterministic & DSP watermark detection layers on raw image bytes."""
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    file_size = len(raw_bytes)

    try:
        img_pil = Image.open(io.BytesIO(raw_bytes))
        img_pil.load()
    except Exception as exc:
        return {
            "filename": filename,
            "error": f"Unable to open image: {exc}",
            "verdict": "ERROR",
            "platform": None,
            "file_info": {"file_size_bytes": file_size, "sha256": sha256},
        }

    width, height = img_pil.size
    img_format = img_pil.format or Path(filename).suffix.lstrip(".").upper() or "UNKNOWN"
    img_mode = img_pil.mode

    layer1 = detect_c2pa(raw_bytes, filename)
    layer2 = detect_pixel_watermarks(img_pil)
    layer3 = detect_container_signatures(raw_bytes, img_pil)
    layer4 = analyze_dsp_carriers(img_pil, include_visuals=include_visuals)

    triggered_layers: list[str] = []
    if layer1["detected"]:
        triggered_layers.append("Layer 1: C2PA / JUMBF")
    if layer2["detected"]:
        triggered_layers.append("Layer 2: Pixel/Freq Bitstream")
    if layer3["detected"]:
        triggered_layers.append("Layer 3: Container/Metadata")
    if layer4["detected"]:
        triggered_layers.append("Layer 4: DSP Carrier")

    if layer1["detected"] or layer2["detected"] or layer3["detected"]:
        verdict = "WATERMARK_CONFIRMED"
    elif layer4["detected"]:
        verdict = "WATERMARK_CARRIER_DETECTED"
    else:
        verdict = "NO_WATERMARK_FOUND"

    platform = (
        layer1.get("platform")
        or layer2.get("platform")
        or layer3.get("platform")
        or layer4.get("carrier_type")
    )

    all_evidence = (
        layer1.get("evidence", [])
        + layer2.get("evidence", [])
        + layer3.get("evidence", [])
        + layer4.get("evidence", [])
    )

    return {
        "filename": filename,
        "verdict": verdict,
        "platform": platform,
        "triggered_layers": triggered_layers,
        "all_evidence": all_evidence,
        "file_info": {
            "format": img_format,
            "width": width,
            "height": height,
            "mode": img_mode,
            "file_size_bytes": file_size,
            "sha256": sha256,
        },
        "layers": {
            "c2pa": layer1,
            "pixel_watermark": layer2,
            "container": layer3,
            "dsp_carrier": layer4,
        },
    }


def analyze_image_file(file_path: str | Path, include_visuals: bool = True) -> dict[str, Any]:
    """Read an image file from disk and run full 4-layer watermark inspection."""
    path = Path(file_path)
    raw_bytes = path.read_bytes()
    return analyze_image_bytes(raw_bytes, filename=path.name, include_visuals=include_visuals)
