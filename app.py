"""Local Flask Web App for Unlimited Non-ML Multi-Layer AI Watermark Detection."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request

from watermark_engine.analyzer import analyze_image_bytes, analyze_image_file
from watermark_engine.sample_generator import generate_demo_samples

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "samples"
STATIC_DIR = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


@app.get("/")
def index() -> object:
    """Serve the main single-page watermark inspector interface."""
    return app.send_static_file("index.html")


@app.post("/api/analyze")
def api_analyze() -> object:
    """Analyze an unlimited batch of uploaded image files without ML classifiers."""
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    results = []
    for storage in files:
        filename = storage.filename or "untitled"
        raw_bytes = storage.read()
        if not raw_bytes:
            results.append(
                {
                    "filename": filename,
                    "verdict": "ERROR",
                    "error": "Empty file (0 bytes).",
                    "platform": None,
                    "triggered_layers": [],
                    "all_evidence": [],
                    "file_info": {"file_size_bytes": 0, "sha256": ""},
                    "layers": {},
                }
            )
            continue
        results.append(analyze_image_bytes(raw_bytes, filename=filename, include_visuals=True))

    return jsonify({"results": results})


@app.post("/api/load-samples")
def api_load_samples() -> object:
    """Generate demo samples if needed and analyze all 8 files across the 4 detection layers."""
    expected_files = (
        "01_dalle3_c2pa.jpg",
        "02_stablediffusion_dwtdctsvd.png",
        "03_novelai_stealth_lsb.png",
        "04_comfyui_workflow.png",
        "05_synthid_frequency_carrier.png",
        "06_clean_camera_control.png",
        "07_flux_48bit_dwtdctsvd.png",
        "08_china_aigc_seedream.png",
    )
    if not all((SAMPLES_DIR / name).is_file() for name in expected_files):
        generate_demo_samples(SAMPLES_DIR)

    results = [
        analyze_image_file(SAMPLES_DIR / name, include_visuals=True)
        for name in expected_files
    ]
    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    print(f"Watermark Inspector running at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
