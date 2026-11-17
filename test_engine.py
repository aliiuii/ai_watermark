"""Automated verification test suite for all 4 non-ML watermark detection layers & multi-provider signatures."""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from watermark_engine.analyzer import analyze_image_bytes, analyze_image_file
from watermark_engine.detectors.c2pa_detector import C2PA_UUID
from watermark_engine.detectors.pixel_watermark_detector import SDXL_48BIT_BYTES, embed_dwtdctsvd
from watermark_engine.sample_generator import _make_base_scene, generate_demo_samples


def _make_png_bytes(img: Image.Image, info: PngImagePlugin.PngInfo | None = None) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _make_c2pa_png_bytes(claim_generator: str) -> bytes:
    rgb = _make_base_scene(128, seed=99)
    raw = _make_png_bytes(Image.fromarray(rgb))
    manifest = json.dumps({"claim_generator": claim_generator}).encode("utf-8")
    payload = C2PA_UUID + b"jumbc2ma" + manifest
    crc = __import__("zlib").crc32(b"caBX" + payload) & 0xFFFFFFFF
    cabx_chunk = struct.pack(">I4s", len(payload), b"caBX") + payload + struct.pack(">I", crc)
    # Insert caBX right before IEND (last 12 bytes)
    return raw[:-12] + cabx_chunk + raw[-12:]


def run_tests() -> None:
    samples_dir = Path(__file__).parent / "samples"
    generate_demo_samples(samples_dir)

    # 1. Layer 1: C2PA / JUMBF in JPEG APP11
    r1 = analyze_image_file(samples_dir / "01_dalle3_c2pa.jpg", include_visuals=True)
    print("[1] 01_dalle3_c2pa.jpg ->", r1["verdict"], "|", r1["platform"], "|", r1["triggered_layers"])
    assert r1["layers"]["c2pa"]["detected"] is True
    assert r1["verdict"] == "WATERMARK_CONFIRMED"
    assert "DALL-E" in (r1["platform"] or "")

    # 2. Layer 2a: DWT-DCT-SVD ("SDV2") with zero metadata
    r2 = analyze_image_file(samples_dir / "02_stablediffusion_dwtdctsvd.png", include_visuals=True)
    print(
        "[2] 02_stablediffusion_dwtdctsvd.png ->",
        r2["verdict"],
        "| payload:",
        r2["layers"]["pixel_watermark"]["decoded_payload"],
        "| consistency:",
        r2["layers"]["pixel_watermark"]["bit_consistency"],
    )
    assert r2["layers"]["pixel_watermark"]["detected"] is True
    assert r2["layers"]["pixel_watermark"]["decoded_payload"] == "SDV2"
    assert r2["verdict"] == "WATERMARK_CONFIRMED"

    # 3. Layer 2b: Stealth PNGInfo Alpha LSB with zero metadata chunks
    r3 = analyze_image_file(samples_dir / "03_novelai_stealth_lsb.png", include_visuals=True)
    print(
        "[3] 03_novelai_stealth_lsb.png ->",
        r3["verdict"],
        "|",
        r3["platform"],
        "| payload len:",
        len(r3["layers"]["pixel_watermark"]["decoded_payload"] or ""),
    )
    assert r3["layers"]["pixel_watermark"]["detected"] is True
    parsed_payload = json.loads(r3["layers"]["pixel_watermark"]["decoded_payload"])
    assert parsed_payload["Software"] == "NovelAI"
    assert r3["verdict"] == "WATERMARK_CONFIRMED"

    # 4. Layer 3: ComfyUI Workflow + IPTC DigitalSourceType chunk
    r4 = analyze_image_file(samples_dir / "04_comfyui_workflow.png", include_visuals=True)
    print("[4] 04_comfyui_workflow.png ->", r4["verdict"], "|", r4["platform"])
    assert r4["layers"]["container"]["detected"] is True
    assert r4["verdict"] == "WATERMARK_CONFIRMED"
    assert r4["platform"] == "ComfyUI"

    # 5. Layer 4: DSP 2D FFT Periodic & Ring Carrier (zero metadata)
    r5 = analyze_image_file(samples_dir / "05_synthid_frequency_carrier.png", include_visuals=True)
    print(
        "[5] 05_synthid_frequency_carrier.png ->",
        r5["verdict"],
        "| metrics:",
        r5["layers"]["dsp_carrier"]["metrics"],
    )
    assert r5["layers"]["dsp_carrier"]["detected"] is True
    assert r5["verdict"] == "WATERMARK_CARRIER_DETECTED"
    assert r5["layers"]["dsp_carrier"]["visuals"]["fft_spectrum_b64"].startswith("data:image/png;base64,")

    # 6. Clean Control Sample
    r6 = analyze_image_file(samples_dir / "06_clean_camera_control.png", include_visuals=True)
    print(
        "[6] 06_clean_camera_control.png ->",
        r6["verdict"],
        "| dsp metrics:",
        r6["layers"]["dsp_carrier"]["metrics"],
    )
    assert r6["verdict"] == "NO_WATERMARK_FOUND"
    assert r6["layers"]["c2pa"]["detected"] is False
    assert r6["layers"]["pixel_watermark"]["detected"] is False
    assert r6["layers"]["container"]["detected"] is False
    assert r6["layers"]["dsp_carrier"]["detected"] is False

    # 7. Layer 2c: Black Forest Labs FLUX 48-bit Frequency Watermark (zero metadata)
    r7 = analyze_image_file(samples_dir / "07_flux_48bit_dwtdctsvd.png", include_visuals=True)
    print("[7] 07_flux_48bit_dwtdctsvd.png ->", r7["verdict"], "|", r7["platform"])
    assert r7["verdict"] == "WATERMARK_CONFIRMED"
    assert "FLUX" in (r7["platform"] or "")

    # 8. Layer 3b: China GB 45438-2025 AIGC Standard (ByteDance Seedream)
    r8 = analyze_image_file(samples_dir / "08_china_aigc_seedream.png", include_visuals=True)
    print("[8] 08_china_aigc_seedream.png ->", r8["verdict"], "|", r8["platform"])
    assert r8["verdict"] == "WATERMARK_CONFIRMED"
    assert "Seedream" in (r8["platform"] or "") or "ByteDance" in (r8["platform"] or "")

    # 9. Additional Provider Unit Checks (SDXL 48-bit, xAI Grok, Canva, Ideogram, SwarmUI, Apple, Samsung)
    sdxl_rgb = embed_dwtdctsvd(_make_base_scene(512, seed=909), payload_bytes=SDXL_48BIT_BYTES, scale=36.0)
    r_sdxl = analyze_image_bytes(_make_png_bytes(Image.fromarray(sdxl_rgb)), "sdxl.png", include_visuals=False)
    assert r_sdxl["verdict"] == "WATERMARK_CONFIRMED" and "SDXL" in (r_sdxl["platform"] or "")

    for provider_str, expected_substr in (
        ("xAI Grok Aurora v2", "Grok"),
        ("Canva Magic Media c2pa-rs/0.31", "Canva"),
        ("Ideogram 3.0", "Ideogram"),
        ("Black Forest Labs FLUX.1-pro", "FLUX"),
        ("Apple Image Playground", "Apple"),
        ("Samsung Galaxy AI", "Samsung"),
        ("Kuaishou Kling AI", "Kling"),
    ):
        res_c2pa = analyze_image_bytes(_make_c2pa_png_bytes(provider_str), "test.png", include_visuals=False)
        assert res_c2pa["verdict"] == "WATERMARK_CONFIRMED", f"Failed C2PA for {provider_str}"
        assert expected_substr in (res_c2pa["platform"] or ""), f"Mismatch {res_c2pa['platform']} for {provider_str}"

    # Container checks for Grok EXIF/PNG & SwarmUI
    grok_info = PngImagePlugin.PngInfo()
    grok_info.add_text("Software", "xAI Grok Imagine")
    r_grok = analyze_image_bytes(
        _make_png_bytes(Image.fromarray(_make_base_scene(128, seed=11)), grok_info),
        "grok.png",
        include_visuals=False,
    )
    assert r_grok["verdict"] == "WATERMARK_CONFIRMED" and "Grok" in (r_grok["platform"] or "")

    swarm_info = PngImagePlugin.PngInfo()
    swarm_info.add_text("parameters", json.dumps({"sui_image_params": {"model": "flux1-dev", "steps": 20}}))
    r_swarm = analyze_image_bytes(
        _make_png_bytes(Image.fromarray(_make_base_scene(128, seed=12)), swarm_info),
        "swarm.png",
        include_visuals=False,
    )
    assert r_swarm["verdict"] == "WATERMARK_CONFIRMED" and r_swarm["platform"] == "SwarmUI"

    print("\nALL 8 SAMPLE TESTS + MULTI-PROVIDER UNIT TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
