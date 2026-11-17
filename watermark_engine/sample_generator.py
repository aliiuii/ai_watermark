"""Deterministic sample generator producing test images for all 4 detection layers."""

from __future__ import annotations

import base64
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from .detectors.c2pa_detector import C2PA_UUID
from .detectors.pixel_watermark_detector import (
    FLUX_48BIT_BYTES,
    embed_dwtdctsvd,
    embed_stealth_pnginfo,
)


def _make_base_scene(size: int = 512, seed: int = 42) -> np.ndarray:
    """Create a rich, smooth procedural RGB scene with natural textures (no ML)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / float(size)

    r = 120 + 75 * np.sin(2.3 * np.pi * x + 0.4) + 35 * np.cos(3.1 * np.pi * y)
    g = 115 + 65 * np.cos(1.9 * np.pi * y + 0.7) + 35 * np.sin(2.7 * np.pi * (x + y))
    b = 135 + 70 * np.sin(2.1 * np.pi * (x - y)) + 30 * np.cos(3.5 * np.pi * x)

    # Mild natural sensor noise
    noise = rng.normal(0.0, 1.4, size=(size, size, 3))
    rgb = np.stack([r, g, b], axis=-1) + noise
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def _inject_jpeg_app11_c2pa(jpeg_bytes: bytes) -> bytes:
    """Inject a valid JPEG APP11 JUMBF superbox carrying OpenAI DALL-E 3 C2PA claims."""
    assert jpeg_bytes.startswith(b"\xff\xd8")
    manifest_json = json.dumps(
        {
            "claim_generator": "OpenAI DALL-E 3 / ChatGPT",
            "softwareAgent": "OpenAI DALL-E 3",
            "assertions": [
                {
                    "label": "c2pa.actions",
                    "data": {
                        "actions": [
                            {
                                "action": "c2pa.created",
                                "digitalSourceType": (
                                    "http://cv.iptc.org/newscodes/digitalsourcetype/"
                                    "trainedAlgorithmicMedia"
                                ),
                            }
                        ]
                    },
                }
            ],
        }
    ).encode("utf-8")

    # Construct JUMBF description + JSON box wrapped inside c2pa/jumb container
    desc_payload = C2PA_UUID + b"\x03c2pa\x00"
    desc_box = struct.pack(">I4s", 8 + len(desc_payload), b"jumd") + desc_payload
    json_box = struct.pack(">I4s", 8 + len(manifest_json), b"json") + manifest_json
    c2ma_box = struct.pack(">I4s", 8 + len(json_box), b"c2ma") + json_box
    jumbf_box = struct.pack(">I4s", 8 + len(desc_box) + len(c2ma_box), b"jumb") + desc_box + c2ma_box

    # JPEG APP11 marker: 0xFF 0xEB + length + CI("JP") + En(1) + Z(1) + payload
    app11_body = b"JP" + struct.pack(">HI", 1, 1) + jumbf_box
    app11_seg = b"\xff\xeb" + struct.pack(">H", len(app11_body) + 2) + app11_body

    return jpeg_bytes[:2] + app11_seg + jpeg_bytes[2:]


def _embed_synthid_style_carrier(rgb: np.ndarray) -> np.ndarray:
    """Embed an imperceptible 2D Fourier periodic lattice + random-phase ring carrier (no metadata)."""
    rng = np.random.default_rng(777)
    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    spec = np.zeros((h, w), dtype=np.complex128)

    # Periodic conjugate Fourier carrier peaks (SynthID/StegaStamp-style synchronization lattice)
    carrier_offsets = [(58, 74), (-58, 74), (92, -46), (112, 84)]
    peak_val = (h * w) * 0.42
    for dy, dx in carrier_offsets:
        spec[cy + dy, cx + dx] = peak_val
        spec[cy - dy, cx - dx] = peak_val

    # Concentric random-phase Fourier ring carrier (Tree-Ring style, avoids (0,0) spatial spike)
    ring_mask = np.abs(radius - 95.0) <= 1.1
    ring_ys, ring_xs = np.where(ring_mask)
    ring_mag = (h * w) * 0.045
    for ry, rx in zip(ring_ys, ring_xs):
        my, mx = 2 * cy - ry, 2 * cx - rx
        if ry < my or (ry == my and rx <= mx):
            phase = rng.uniform(0, 2 * np.pi)
            val = ring_mag * np.exp(1j * phase)
            spec[ry, rx] += val
            if 0 <= my < h and 0 <= mx < w and (my != ry or mx != rx):
                spec[my, mx] += np.conj(val)

    spatial_carrier = np.real(np.fft.ifft2(np.fft.ifftshift(spec)))
    # Keep perturbation imperceptible (std ~ 1.8 pixel levels, PSNR > 43 dB)
    spatial_carrier = spatial_carrier / (np.std(spatial_carrier) + 1e-8) * 1.8

    out = rgb.astype(np.float64)
    out[:, :, 0] += spatial_carrier
    out[:, :, 1] += spatial_carrier
    out[:, :, 2] += spatial_carrier
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def generate_demo_samples(output_dir: str | Path) -> list[dict[str, str]]:
    """Generate 8 realistic test images covering all 4 non-ML detection layers + 1 clean control."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, str]] = []

    # 1. Layer 1: DALL-E 3 C2PA / JUMBF in JPEG APP11
    rgb1 = _make_base_scene(512, seed=101)
    buf = io.BytesIO()
    Image.fromarray(rgb1).save(buf, format="JPEG", quality=92)
    c2pa_jpeg = _inject_jpeg_app11_c2pa(buf.getvalue())
    p1 = out_path / "01_dalle3_c2pa.jpg"
    p1.write_bytes(c2pa_jpeg)
    generated.append({"file": p1.name, "layer": "Layer 1 (C2PA / JUMBF)"})

    # 2. Layer 2a: Stable Diffusion DWT-DCT-SVD ("SDV2") with ZERO metadata
    rgb2 = _make_base_scene(512, seed=202)
    watermarked_sd = embed_dwtdctsvd(rgb2, payload_bytes=b"SDV2", scale=36.0)
    p2 = out_path / "02_stablediffusion_dwtdctsvd.png"
    Image.fromarray(watermarked_sd).save(p2, format="PNG")
    generated.append({"file": p2.name, "layer": "Layer 2 (DWT-DCT-SVD 'SDV2')"})

    # 3. Layer 2b: NovelAI Stealth PNGInfo Alpha LSB with ZERO metadata chunks
    rgb3 = _make_base_scene(512, seed=303)
    rgba3 = np.dstack([rgb3, np.full((512, 512), 255, dtype=np.uint8)])
    stealth_payload = json.dumps(
        {
            "Software": "NovelAI",
            "prompt": "cyberpunk botanical conservatory, volumetric light, 8k",
            "uc": "lowres, bad anatomy, watermark",
            "steps": 28,
            "sampler": "k_euler_ancestral",
            "seed": 194857234,
        }
    )
    stealth_rgba = embed_stealth_pnginfo(rgba3, stealth_payload, compressed=False)
    p3 = out_path / "03_novelai_stealth_lsb.png"
    Image.fromarray(stealth_rgba).save(p3, format="PNG")
    generated.append({"file": p3.name, "layer": "Layer 2 (Stealth LSB PNGInfo)"})

    # 4. Layer 3: ComfyUI workflow + IPTC DigitalSourceType PNG tEXt chunk
    rgb4 = _make_base_scene(512, seed=404)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text(
        "prompt",
        json.dumps(
            {
                "3": {
                    "class_type": "KSampler",
                    "inputs": {"seed": 88442211, "steps": 25, "cfg": 7.0, "sampler_name": "dpmpp_2m"},
                },
                "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-dev.safetensors"}},
            }
        ),
    )
    png_info.add_text("DigitalSourceType", "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia")
    p4 = out_path / "04_comfyui_workflow.png"
    Image.fromarray(rgb4).save(p4, format="PNG", pnginfo=png_info)
    generated.append({"file": p4.name, "layer": "Layer 3 (ComfyUI & IPTC Chunks)"})

    # 5. Layer 4: Synthetic SynthID/Tree-Ring-style Fourier Frequency Carrier (zero metadata)
    rgb5 = _make_base_scene(512, seed=505)
    carrier_rgb = _embed_synthid_style_carrier(rgb5)
    p5 = out_path / "05_synthid_frequency_carrier.png"
    Image.fromarray(carrier_rgb).save(p5, format="PNG")
    generated.append({"file": p5.name, "layer": "Layer 4 (DSP 2D FFT & Ring Carrier)"})

    # 6. Clean Control Image (No watermark)
    rgb6 = _make_base_scene(512, seed=606)
    p6 = out_path / "06_clean_camera_control.png"
    Image.fromarray(rgb6).save(p6, format="PNG")
    generated.append({"file": p6.name, "layer": "Control (Clean / No Watermark)"})

    # 7. Layer 2c: Black Forest Labs FLUX 48-bit Frequency Watermark (zero metadata)
    rgb7 = _make_base_scene(512, seed=707)
    flux_wm = embed_dwtdctsvd(rgb7, payload_bytes=FLUX_48BIT_BYTES, scale=36.0)
    p7 = out_path / "07_flux_48bit_dwtdctsvd.png"
    Image.fromarray(flux_wm).save(p7, format="PNG")
    generated.append({"file": p7.name, "layer": "Layer 2 (Black Forest Labs FLUX 48-bit)"})

    # 8. Layer 3b: China AIGC GB 45438-2025 Standard (ByteDance Seedream Base64 JSON)
    rgb8 = _make_base_scene(512, seed=808)
    aigc_json = json.dumps(
        {
            "AIGC": {
                "Label": "1",
                "ContentProducer": "001101100000000000_ByteDance_Seedream_3.0",
                "ProduceID": "sdrm-2026-99887766",
                "ReservedCode1": "",
                "ContentPropagator": "001101100000000000_Douyin",
                "PropagateID": "prop-11223344",
            }
        }
    ).encode("utf-8")
    aigc_info = PngImagePlugin.PngInfo()
    aigc_info.add_text("AIGC", base64.b64encode(aigc_json).decode("ascii"))
    p8 = out_path / "08_china_aigc_seedream.png"
    Image.fromarray(rgb8).save(p8, format="PNG", pnginfo=aigc_info)
    generated.append({"file": p8.name, "layer": "Layer 3 (GB 45438-2025 AIGC / Seedream)"})

    return generated
