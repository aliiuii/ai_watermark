# Watermark Inspector

A local, **non-machine-learning** multi-layer inspector for detecting hidden AI watermarks, content credentials, embedded bitstreams, metadata provenance, and frequency-domain carriers.

Built for unlimited local batch scanning via both a **Web UI** and **CLI**, with zero cloud API limits, zero GPU requirements, and zero ML image-anomaly classifiers.

---

## Why Non-ML?

Most AI-image detectors train neural classifiers on visual artifacts (blur, noise smoothness, textures), which suffer from high false-positive rates on edited photos and cannot extract concrete forensic proof. **Watermark Inspector** inspects the actual bytes, bitstreams, metadata containers, and frequency spectra of an image:

- **Layers 1–3 (Deterministic)** extract verifiable records directly from JUMBF boxes, pixel/wavelet coefficients, LSB streams, or binary chunks.
- **Layer 4 (DSP Signal Analysis)** uses deterministic Digital Signal Processing (2D FFT, radial ring profiling, Haar `HH` wavelet kurtosis) to surface hidden spectral carriers when metadata has been stripped.

---

## Detection Pipeline (4 Layers)

```text
Image Bytes (PNG / JPEG / WebP / BMP / TIFF / AVIF)
  ├── Layer 1: C2PA / JUMBF Scanner       (APP11, caBX, C2PA UUID, claim_generator)
  ├── Layer 2: Pixel & Freq Bitstream     (DWT-DCT-SVD, DWT-DCT, Alpha/RGB Stealth LSB)
  ├── Layer 3: Deep Container & Metadata  (PNG chunks, EXIF, XMP, IPTC, China GB 45438-2025 AIGC)
  └── Layer 4: DSP Spectrum Carrier       (2D FFT local Z-score peaks, radial ring delta, Haar HH)
```

| Layer | Module | What It Extracts / Measures | Verdict Level |
| :--- | :--- | :--- | :--- |
| **Layer 1** | [`c2pa_detector.py`](watermark_engine/detectors/c2pa_detector.py) | JUMBF boxes (`jumb`, `c2pa`, `c2ma`, `c2cl`, `c2cs`, `caBX`), C2PA Manifest UUID `d8fec3d6-1b0e-483c-9297-5828877ec481`, `claim_generator` / `softwareAgent`, `c2pa.actions`, IPTC `digitalSourceType` | `WATERMARK_CONFIRMED` |
| **Layer 2** | [`pixel_watermark_detector.py`](watermark_engine/detectors/pixel_watermark_detector.py) | `stealth_pnginfo` / `stealth_pngcomp` / `stealth_rgbinfo` / `stealth_rgbcomp` in `(x,y)` & `(y,x)` order; 2D Haar DWT + 4×4 DCT + SVD/DC QIM bitstreams (`SDV2`, `StableDiffusionV1`, 48-bit FLUX `0x2AFE879CF52E`, 48-bit SDXL `0xB3EC907BB19E`) | `WATERMARK_CONFIRMED` |
| **Layer 3** | [`container_detector.py`](watermark_engine/detectors/container_detector.py) | Raw PNG `tEXt`/`zTXt`/`iTXt` chunks, JPEG `APP0`–`APP15`/`COM`, WebP RIFF chunks, EXIF, XMP packets, IPTC `DigitalSourceType`, ComfyUI node graphs, A1111/Forge/Fooocus/SwarmUI/InvokeAI/NovelAI params, and China **GB 45438-2025 AIGC** JSON/Base64 records | `WATERMARK_CONFIRMED` |
| **Layer 4** | [`dsp_carrier_detector.py`](watermark_engine/detectors/dsp_carrier_detector.py) | High-pass residual 2D FFT spectral peak Z-score, concentric radial ring modulation (Tree-Ring style), Haar `HH` kurtosis, and 4×4 SVD QIM phase clustering + diagnostic PNG visualizations | `WATERMARK_CARRIER_DETECTED` |

---

## Supported Providers & Standards

- **C2PA / Content Credentials (`Layer 1`)**: OpenAI (ChatGPT / DALL-E / Sora), Black Forest Labs (FLUX), Google (Gemini / Imagen / Veo / SynthID C2PA), Adobe (Firefly / Stock / Photoshop), Microsoft (Designer / Bing Image Creator / Copilot), Canva (Magic Media), Ideogram, Recraft AI, Leonardo AI, Meta AI, Midjourney, Stability AI, Runway AI, Luma Labs, Apple (Image Playground / Apple Intelligence), Samsung Galaxy AI, xAI (Grok / Aurora), ByteDance, Alibaba, Kuaishou, Tencent, MiniMax, Freepik.
- **Invisible Pixel & Frequency Bitstreams (`Layer 2`)**:
  - **Black Forest Labs FLUX**: 48-bit frequency signature (`0x2AFE879CF52E`)
  - **Stability AI SDXL (Diffusers)**: 48-bit frequency signature (`0xB3EC907BB19E`)
  - **Stability AI SD v2 / v1**: `SDV2` (32-bit), `StableDiffusionV1` (136-bit)
  - **Stealth LSB PNGInfo**: NovelAI, SD WebUI / Forge Stealth extension, Fooocus, SwarmUI, ComfyUI, Draw Things
- **Container Metadata & Regulatory Standards (`Layer 3`)**:
  - **Local UIs**: ComfyUI (`prompt`/`workflow`), SwarmUI (`sui_image_params`), Automatic1111 / Forge / Fooocus (`parameters`), InvokeAI (`sd-metadata`), NovelAI, Draw Things
  - **IPTC / XMP Provenance**: `trainedAlgorithmicMedia`, `compositeWithTrainedAlgorithmicMedia`, `algorithmicMedia`, and vendor XMP/EXIF identifiers (including Civitai, Tensor.art, SeaArt, xAI/Grok, Apple, Samsung, Canva, Ideogram, Recraft, Leonardo AI)
  - **China AIGC Mandatory Standard (`GB 45438-2025`)**: Plain or Base64-encoded `AIGC` JSON (`Label`, `ContentProducer`, `ProduceID`, `ContentPropagator`, `PropagateID`) used by ByteDance (Seedream / Jimeng / Doubao), Alibaba (Qwen / Tongyi / Wanx), Kuaishou (Kling / Kolors), Tencent (Hunyuan), Baidu, MiniMax (Hailuo), and Zhipu.

---

## Installation & Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.11.9)

### Setup

```bash
git clone https://github.com/<your-username>/ai-watermark-inspector.git
cd ai-watermark-inspector
pip install -r requirements.txt
```

### Run the Web App

```bash
python app.py
```

Open `http://127.0.0.1:5050` in your browser (or set `PORT=8080 python app.py`).
- Drag and drop any number of images into the upload area, or click **Load Demo Samples** to generate and inspect 8 built-in samples across all 4 layers.
- Click **Details** on any row to inspect C2PA data, decoded pixel payloads, raw container chunks, DSP metrics, and the **2D FFT Spectrum**, **LSB Bit-Plane**, and **High-Pass Residual (20x)** maps.
- Export results via **Export JSON** or **Export CSV**.

### Run the CLI (Batch Scanning)

```bash
# Generate 8 synthetic demo samples in ./samples
python cli.py generate-samples --out samples

# Scan a single file or recursively scan a directory
python cli.py scan samples

# Export full JSON and summary CSV reports
python cli.py scan /path/to/images --json report.json --csv report.csv
```

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.avif`.

### Run the Test Suite

```bash
python test_engine.py
```

Generates the 8 demo samples and runs deterministic assertions across Layers 1–4 plus multi-provider unit checks (`SDXL 48-bit`, `xAI Grok`, `Canva`, `Ideogram`, `FLUX`, `Apple Image Playground`, `Samsung Galaxy AI`, `Kuaishou Kling`, and `SwarmUI`).

---

## REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the local Web UI (`static/index.html`) |
| `POST` | `/api/analyze` | Multipart upload (`files` field, unlimited batch) returning `{ "results": [...] }` with full layer reports and Base64 diagnostic PNGs |
| `POST` | `/api/load-samples` | Generates (if missing) and analyzes the 8 demo test samples |

---

## Strengths

- **Zero ML / Zero Rate Limits**: Completely offline and deterministic; no API keys, model weights, or GPU needed.
- **Actionable Forensic Evidence**: Returns exact decoded text/bitstreams, raw chunk names, JSON workflows, C2PA generator fields, SHA-256 hashes, and frequency spectrum images rather than an opaque probability score.
- **Multi-Layer Coverage**: Even if metadata is absent (like DWT-DCT-SVD or Stealth LSB images, or frequency carriers), Layers 2 and 4 inspect the pixel and spectral domains directly.
- **No Frequency Resampling Distortion**: Layer 4 center-crops up to `512×512` at native 1:1 pixel resolution with a smooth Tukey window instead of bilinear resizing, preserving high-frequency carrier peaks.
- **Safe Informational EXIF Signal**: Records `camera_exif_present` (`Present` / `Absent`) as contextual metadata without falsely flagging clean PNGs or screenshots as AI-generated.

---

## Limitations & Threat Model

Understand these technical boundaries before relying on results in production or moderation pipelines:

1. **Metadata Stripping by Social Platforms & Screenshots**:
   Uploading images to X/Twitter, Instagram, Facebook, WhatsApp, or taking screenshots strips JPEG `APP11` (C2PA), PNG chunks, EXIF, and XMP headers. Layers 1 and 3 cannot recover stripped containers.
2. **Fragility of Pixel & Block Watermarks**:
   - **Stealth LSB** (`stealth_pnginfo`) is fragile and destroyed by lossy JPEG re-encoding, resizing, or alpha-channel removal.
   - **DWT-DCT / DWT-DCT-SVD** (`SDV2`, `FLUX`, `SDXL`) tolerates mild encoding noise, but heavy JPEG compression, cropping that misaligns the `4×4` DCT grid, or geometric scaling prevents deterministic decoding.
3. **C2PA Structural Inspection vs. PKI Signature Verification**:
   Layer 1 parses JUMBF containers, C2PA UUIDs, and manifest strings locally without validating X.509 certificate chains against an external C2PA Trust List. For cryptographic chain-of-trust validation against tampered certificates, pair with the official [`c2patool`](https://github.com/contentauth/c2patool).
4. **Closed-Key Neural Watermarks (e.g., Proprietary SynthID / StegaStamp)**:
   Closed-key watermarks that embed keys held exclusively by the vendor cannot be decoded into readable payloads offline. Layer 4 detects statistical anomalies in 2D FFT spectral peaks and radial ring energy left by periodic/ring carriers, which can also be triggered by natural periodic textures (e.g., moiré patterns, brick grids, fine striped fabrics).
5. **Providers Relying Only on Visible Watermarks (e.g., xAI Grok / Aurora)**:
   Standard exports from generators such as **xAI Grok / Aurora** currently rely primarily on a visible corner watermark and emit "metadata-naked" files without C2PA or hidden pixel watermarks. If the visible corner mark is cropped out and no container metadata was written, no hidden deterministic watermark remains to extract.
6. **Non-DC `dwtDct` Permutation Variants**:
   Layer 2 checks standard DWT-DCT-SVD (`s[0]`) and DWT-DCT (`dct[0,0]`) QIM bitstreams; third-party variants that permute arbitrary AC coefficients with custom seeds require the corresponding decoder key.

---

## Project Structure

```text
.
├── app.py                              # Flask web server and REST API
├── cli.py                              # Unlimited batch CLI scanner & JSON/CSV exporter
├── test_engine.py                      # Automated multi-layer & multi-provider test suite
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── static/
│   ├── index.html                      # Single-page UI
│   ├── styles.css                      # Responsive container table, modal, and tabs CSS
│   └── app.js                          # Batch upload, dialog fallbacks, and chart rendering
└── watermark_engine/
    ├── __init__.py
    ├── analyzer.py                     # Unified 4-layer orchestration & verdict logic
    ├── sample_generator.py             # Generates 8 synthetic test samples
    └── detectors/
        ├── __init__.py
        ├── c2pa_detector.py            # Layer 1: C2PA / JUMBF binary box scanner
        ├── pixel_watermark_detector.py # Layer 2: DWT-DCT-SVD, DWT-DCT & Stealth LSB decoders
        ├── container_detector.py       # Layer 3: PNG/JPEG/WebP chunks, EXIF, XMP & GB 45438-2025
        └── dsp_carrier_detector.py     # Layer 4: 2D FFT, radial ring & Haar HH DSP analyzer
```

---

## License

Released under the [MIT License](LICENSE).
