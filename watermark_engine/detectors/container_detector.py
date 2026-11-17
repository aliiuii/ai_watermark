"""Layer 3: Deep Container, Chunk & Metadata Signature Parser (PNG/JPEG/WebP/XMP/IPTC/EXIF/AIGC)."""

from __future__ import annotations

import base64
import json
import re
import struct
import zlib
from typing import Any

from PIL import ExifTags, Image

_XMP_RE = re.compile(rb"<x:xmpmeta[\s\S]*?</x:xmpmeta>", re.IGNORECASE)
_CAMERA_EXIF_KEYS = frozenset({"Make", "Model", "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength"})

# Deterministic IPTC / XMP / EXIF provider patterns (pattern, label, description, word_boundary)
PROVIDER_METADATA_RULES: tuple[tuple[str, str, str, bool], ...] = (
    ("trainedalgorithmicmedia", "IPTC Trained Algorithmic Media", "IPTC DigitalSourceType: trainedAlgorithmicMedia", False),
    (
        "compositewithtrainedalgorithmicmedia",
        "IPTC Composite Trained Algorithmic Media",
        "IPTC DigitalSourceType: compositeWithTrainedAlgorithmicMedia",
        False,
    ),
    ("algorithmicmedia", "IPTC Algorithmic Media", "IPTC DigitalSourceType: algorithmicMedia", False),
    ("made with google ai", "Google AI / SynthID", "XMP/IPTC credit: Made with Google AI", False),
    ("synthid", "Google SynthID", "Google SynthID metadata tag", False),
    ("google gemini", "Google Gemini", "Google Gemini metadata signature", False),
    ("google imagen", "Google Imagen", "Google Imagen metadata signature", False),
    ("adobe firefly", "Adobe Firefly", "Adobe Firefly metadata signature", False),
    ("dall-e", "OpenAI DALL-E", "OpenAI DALL-E metadata signature", False),
    ("chatgpt", "OpenAI ChatGPT / GPT-4o", "OpenAI ChatGPT metadata signature", False),
    ("openai", "OpenAI", "OpenAI metadata signature", False),
    ("bing image creator", "Microsoft Bing Image Creator", "Microsoft Bing Image Creator signature", False),
    ("microsoft designer", "Microsoft Designer", "Microsoft Designer signature", False),
    ("black forest labs", "Black Forest Labs (FLUX)", "Black Forest Labs (FLUX) signature", False),
    ("flux.1", "Black Forest Labs (FLUX)", "FLUX.1 model identifier in metadata", False),
    ("flux.2", "Black Forest Labs (FLUX)", "FLUX.2 model identifier in metadata", False),
    ("grok imagine", "xAI Grok Imagine", "xAI Grok Imagine metadata signature", False),
    ("xai", "xAI Grok / Aurora", "xAI identifier in container metadata", True),
    ("grok", "xAI Grok / Aurora", "Grok identifier in container metadata", True),
    ("canva", "Canva AI / Magic Media", "Canva AI metadata signature", True),
    ("ideogram", "Ideogram", "Ideogram metadata signature", True),
    ("recraft", "Recraft AI", "Recraft AI metadata signature", True),
    ("leonardo.ai", "Leonardo AI", "Leonardo.ai metadata signature", False),
    ("leonardo ai", "Leonardo AI", "Leonardo AI metadata signature", False),
    ("meta ai", "Meta AI", "Meta AI metadata signature", False),
    ("imagine with meta", "Meta AI", "Imagine with Meta metadata signature", False),
    ("midjourney", "Midjourney", "Midjourney metadata signature", False),
    ("runwayml", "Runway AI", "RunwayML metadata signature", False),
    ("luma labs", "Luma Labs / Dream Machine", "Luma Labs metadata signature", False),
    ("image playground", "Apple Image Playground", "Apple Image Playground metadata signature", False),
    ("apple intelligence", "Apple Intelligence", "Apple Intelligence metadata signature", False),
    ("galaxy ai", "Samsung Galaxy AI", "Samsung Galaxy AI metadata signature", False),
    ("draw things", "Draw Things (SD / FLUX)", "Draw Things metadata signature", False),
    ("seedream", "ByteDance Seedream", "ByteDance Seedream metadata signature", False),
    ("jimeng", "ByteDance Jimeng", "ByteDance Jimeng metadata signature", False),
    ("doubao", "ByteDance Doubao", "ByteDance Doubao metadata signature", False),
    ("tongyi", "Alibaba Tongyi Wanxiang", "Alibaba Tongyi metadata signature", False),
    ("wanx", "Alibaba Wanx", "Alibaba Wanx metadata signature", True),
    ("qwen", "Alibaba Qwen", "Alibaba Qwen metadata signature", True),
    ("kling", "Kuaishou Kling", "Kuaishou Kling metadata signature", True),
    ("kolors", "Kuaishou Kolors", "Kuaishou Kolors metadata signature", True),
    ("hunyuan", "Tencent Hunyuan", "Tencent Hunyuan metadata signature", False),
    ("hailuo", "MiniMax Hailuo", "MiniMax Hailuo metadata signature", False),
    ("minimax", "MiniMax AI", "MiniMax AI metadata signature", True),
    ("freepik pikaso", "Freepik Pikaso", "Freepik Pikaso metadata signature", False),
    ("civitai", "Civitai", "Civitai metadata signature", False),
    ("tensor.art", "Tensor.art", "Tensor.art metadata signature", False),
    ("seaart", "SeaArt AI", "SeaArt AI metadata signature", False),
)

CHINA_AIGC_PRODUCER_MAP: tuple[tuple[str, str], ...] = (
    ("bytedance", "ByteDance (Seedream / Jimeng / Doubao)"),
    ("seedream", "ByteDance Seedream"),
    ("jimeng", "ByteDance Jimeng"),
    ("alibaba", "Alibaba (Qwen / Tongyi Wanxiang)"),
    ("tongyi", "Alibaba Tongyi Wanxiang"),
    ("qwen", "Alibaba Qwen"),
    ("kuaishou", "Kuaishou (Kling / Kolors)"),
    ("kling", "Kuaishou Kling"),
    ("tencent", "Tencent (Hunyuan)"),
    ("hunyuan", "Tencent Hunyuan"),
    ("baidu", "Baidu (ERNIE / Yige)"),
    ("minimax", "MiniMax (Hailuo)"),
    ("zhipu", "Zhipu AI"),
)


def _parse_png_chunks(raw_bytes: bytes) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Parse PNG chunks directly from raw bytes, decoding tEXt, zTXt, and iTXt payloads."""
    if not raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return [], {}

    chunks: list[dict[str, Any]] = []
    text_kv: dict[str, str] = {}
    pos = 8
    total = len(raw_bytes)

    while pos + 12 <= total:
        chunk_len = struct.unpack(">I", raw_bytes[pos : pos + 4])[0]
        chunk_type_bytes = raw_bytes[pos + 4 : pos + 8]
        chunk_type = chunk_type_bytes.decode("ascii", errors="replace")
        data_start = pos + 8
        data_end = data_start + chunk_len
        if data_end + 4 > total:
            break
        data = raw_bytes[data_start:data_end]

        chunk_info: dict[str, Any] = {
            "type": chunk_type,
            "length": chunk_len,
            "offset": pos,
        }

        if chunk_type == "tEXt" and b"\x00" in data:
            key_b, val_b = data.split(b"\x00", 1)
            key = key_b.decode("latin1", errors="replace")
            val = val_b.decode("utf-8", errors="replace")
            text_kv[key] = val
            chunk_info["key"] = key
            chunk_info["preview"] = val[:240]
        elif chunk_type == "zTXt" and b"\x00" in data:
            key_b, rest = data.split(b"\x00", 1)
            key = key_b.decode("latin1", errors="replace")
            if len(rest) >= 2:
                try:
                    val = zlib.decompress(rest[1:]).decode("utf-8", errors="replace")
                    text_kv[key] = val
                    chunk_info["key"] = key
                    chunk_info["preview"] = val[:240]
                except Exception:
                    chunk_info["key"] = key
                    chunk_info["preview"] = "<zlib decode error>"
        elif chunk_type == "iTXt" and b"\x00" in data:
            parts = data.split(b"\x00", 5)
            if len(parts) >= 6:
                key = parts[0].decode("utf-8", errors="replace")
                comp_flag = parts[1][:1]
                raw_val = parts[5]
                try:
                    if comp_flag == b"\x01":
                        val = zlib.decompress(raw_val).decode("utf-8", errors="replace")
                    else:
                        val = raw_val.decode("utf-8", errors="replace")
                except Exception:
                    val = raw_val.decode("utf-8", errors="replace")
                text_kv[key] = val
                chunk_info["key"] = key
                chunk_info["preview"] = val[:240]

        chunks.append(chunk_info)
        if chunk_type == "IEND":
            break
        pos = data_end + 4

    return chunks, text_kv


def _parse_jpeg_segments(raw_bytes: bytes) -> list[dict[str, Any]]:
    """List JPEG marker segments (APP0..APP15, COM) with sizes and offsets."""
    if not raw_bytes.startswith(b"\xff\xd8"):
        return []
    segments: list[dict[str, Any]] = []
    pos = 2
    total = len(raw_bytes)
    while pos + 4 <= total:
        if raw_bytes[pos] != 0xFF:
            break
        marker = raw_bytes[pos + 1]
        if marker in (0xD8, 0xD9):
            pos += 2
            continue
        if marker == 0xDA:
            segments.append({"type": "SOS", "offset": pos, "length": total - pos})
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2
            continue
        seg_len = struct.unpack(">H", raw_bytes[pos + 2 : pos + 4])[0]
        if seg_len < 2 or pos + 2 + seg_len > total:
            break
        name = (
            f"APP{marker - 0xE0}"
            if 0xE0 <= marker <= 0xEF
            else ("COM" if marker == 0xFE else f"0xFF{marker:02X}")
        )
        seg_data = raw_bytes[pos + 4 : pos + 2 + seg_len]
        preview = ""
        if marker == 0xFE or (0xE0 <= marker <= 0xEF):
            preview = seg_data[:64].decode("latin1", errors="replace").replace("\x00", " ").strip()
        segments.append({"type": name, "offset": pos, "length": seg_len, "preview": preview[:120]})
        pos += 2 + seg_len
    return segments


def _parse_webp_chunks(raw_bytes: bytes) -> list[dict[str, Any]]:
    """List RIFF chunks inside a WebP file."""
    if len(raw_bytes) < 12 or raw_bytes[:4] != b"RIFF" or raw_bytes[8:12] != b"WEBP":
        return []
    chunks: list[dict[str, Any]] = []
    pos = 12
    total = len(raw_bytes)
    while pos + 8 <= total:
        fourcc = raw_bytes[pos : pos + 4].decode("ascii", errors="replace")
        size = struct.unpack("<I", raw_bytes[pos + 4 : pos + 8])[0]
        chunks.append({"type": fourcc, "offset": pos, "length": size})
        pos += 8 + size + (size % 2)
    return chunks


def _extract_exif_dict(img_pil: Image.Image) -> dict[str, str]:
    """Read human-readable EXIF tags from Pillow Image."""
    exif_dict: dict[str, str] = {}
    try:
        raw_exif = img_pil.getexif()
        if not raw_exif:
            return exif_dict
        for tag_id, value in raw_exif.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if isinstance(value, bytes):
                try:
                    decoded = value.decode("utf-8", errors="replace").strip("\x00")
                except Exception:
                    decoded = repr(value[:64])
            else:
                decoded = str(value)
            exif_dict[str(tag_name)] = decoded[:1000]
    except Exception:
        pass
    return exif_dict


def _extract_xmp_packets(raw_bytes: bytes) -> list[str]:
    """Extract embedded XMP XML packets from any binary container."""
    return [m.decode("utf-8", errors="replace") for m in _XMP_RE.findall(raw_bytes)]


def _looks_like_comfyui(val: str) -> bool:
    """Detect ComfyUI node graph JSON in prompt/workflow strings."""
    try:
        parsed = json.loads(val)
    except Exception:
        return False
    if isinstance(parsed, dict):
        if "nodes" in parsed and "links" in parsed:
            return True
        for node in parsed.values():
            if isinstance(node, dict) and "class_type" in node and "inputs" in node:
                return True
    return False


def _try_parse_aigc_json(candidate: str) -> dict[str, Any] | None:
    """Parse GB 45438-2025 China AIGC JSON metadata (raw JSON or Base64-encoded JSON)."""
    text = candidate.strip()
    if not text:
        return None

    for attempt in (text,):
        if attempt.startswith("{") and ("ContentProducer" in attempt or '"AIGC"' in attempt):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    aigc = obj.get("AIGC", obj)
                    if isinstance(aigc, dict) and ("ContentProducer" in aigc or "ProduceID" in aigc):
                        return aigc
            except Exception:
                pass

    # Base64-wrapped AIGC payload (e.g. XmpKvBase64 / PNG AIGC chunk)
    if len(text) >= 24 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", text):
        try:
            decoded = base64.b64decode(text, validate=True).decode("utf-8", errors="ignore").strip()
            if decoded.startswith("{") and ("ContentProducer" in decoded or '"AIGC"' in decoded):
                obj = json.loads(decoded)
                if isinstance(obj, dict):
                    aigc = obj.get("AIGC", obj)
                    if isinstance(aigc, dict) and ("ContentProducer" in aigc or "ProduceID" in aigc):
                        return aigc
        except Exception:
            pass
    return None


def _extract_china_aigc(merged_texts: dict[str, str], exif_tags: dict[str, str], xmp_packets: list[str]) -> dict[str, Any] | None:
    """Search metadata containers for GB 45438-2025 AIGC metadata records."""
    candidates: list[str] = []
    for key, val in merged_texts.items():
        if key.lower() in ("aigc", "xmpkvbase64", "comment", "description") or "ContentProducer" in val:
            candidates.append(val)
    for val in exif_tags.values():
        if "ContentProducer" in val or "AIGC" in val:
            candidates.append(val)
    for pkt in xmp_packets:
        if "ContentProducer" in pkt or "AIGC" in pkt:
            # Check raw JSON blocks or Base64 runs inside XMP
            for match in re.findall(r"\{[^{}]*ContentProducer[^{}]*\}", pkt):
                candidates.append(match)
            for b64_run in re.findall(r"[A-Za-z0-9+/]{32,}={0,2}", pkt):
                candidates.append(b64_run)

    for cand in candidates:
        parsed = _try_parse_aigc_json(cand)
        if parsed is not None:
            return parsed
    return None


def _matches_rule(text_lower: str, needle: str, word_boundary: bool) -> bool:
    """Check substring or word-boundary match cleanly."""
    if not word_boundary:
        return needle in text_lower
    return re.search(rf"\b{re.escape(needle)}\b", text_lower) is not None


def detect_container_signatures(raw_bytes: bytes, img_pil: Image.Image) -> dict[str, Any]:
    """Inspect container chunks, text metadata, EXIF, IPTC, XMP, and AIGC tags."""
    png_chunks, png_texts = _parse_png_chunks(raw_bytes)
    jpeg_segments = _parse_jpeg_segments(raw_bytes)
    webp_chunks = _parse_webp_chunks(raw_bytes)
    chunks = png_chunks or jpeg_segments or webp_chunks

    pil_info: dict[str, str] = {}
    for k, v in (img_pil.info or {}).items():
        if isinstance(v, (str, bytes)):
            val_str = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v
            if k.lower() not in ("icc_profile",):
                pil_info[str(k)] = val_str[:4000]

    merged_texts = {**pil_info, **png_texts}
    exif_tags = _extract_exif_dict(img_pil)
    xmp_packets = _extract_xmp_packets(raw_bytes)
    camera_exif_present = any(k in exif_tags for k in _CAMERA_EXIF_KEYS)

    evidence: list[str] = []
    platforms: list[str] = []

    # 1. ComfyUI
    for key in ("prompt", "workflow"):
        if key in merged_texts and _looks_like_comfyui(merged_texts[key]):
            platforms.append("ComfyUI")
            evidence.append(f"ComfyUI node graph in '{key}' chunk")
            break

    # 2. SwarmUI
    if (
        "sui_image_params" in merged_texts
        or "sui_image_params" in merged_texts.get("parameters", "")
        or "sui_image_params" in exif_tags.get("UserComment", "")
    ):
        platforms.append("SwarmUI")
        evidence.append("SwarmUI parameters ('sui_image_params') found")

    # 3. Fooocus / Automatic1111 / Forge
    params_text = merged_texts.get("parameters", "") or exif_tags.get("UserComment", "")
    if params_text and ("Steps:" in params_text or "Sampler:" in params_text or "CFG scale:" in params_text):
        if "Fooocus" in params_text or merged_texts.get("Software", "").lower() == "fooocus":
            platforms.append("Fooocus")
            evidence.append("Fooocus generation parameters found")
        else:
            platforms.append("Automatic1111 / Forge / SD WebUI")
            evidence.append("Stable Diffusion generation parameters ('Steps:', 'Sampler:', 'CFG scale:') found")

    # 4. InvokeAI
    if "sd-metadata" in merged_texts or "invokeai_metadata" in merged_texts:
        platforms.append("InvokeAI")
        evidence.append("InvokeAI metadata chunk found")

    # 5. NovelAI
    if merged_texts.get("Software", "").lower() == "novelai" or (
        "Comment" in merged_texts and '"uc"' in merged_texts["Comment"] and '"sampler"' in merged_texts["Comment"].lower()
    ):
        platforms.append("NovelAI")
        evidence.append("NovelAI PNG metadata signature found")

    # 6. GB 45438-2025 China AIGC Metadata Standard
    aigc_record = _extract_china_aigc(merged_texts, exif_tags, xmp_packets)
    if aigc_record is not None:
        producer_str = str(aigc_record.get("ContentProducer", ""))
        producer_lower = producer_str.lower()
        mapped_cn = "China AIGC Standard (GB 45438-2025)"
        for needle, label in CHINA_AIGC_PRODUCER_MAP:
            if needle in producer_lower:
                mapped_cn = label
                break
        platforms.append(mapped_cn)
        evidence.append(
            f"GB 45438-2025 AIGC metadata (Label={aigc_record.get('Label')}, "
            f"ContentProducer={producer_str[:64]}, ProduceID={aigc_record.get('ProduceID', '-')})"
        )

    # 7. Midjourney Job ID
    all_text_lower = (
        "\n".join(merged_texts.values())
        + "\n"
        + "\n".join(exif_tags.values())
        + "\n"
        + "\n".join(xmp_packets)
    ).lower()

    if "job id:" in all_text_lower and ("midjourney" in all_text_lower or "digitalimageguid" in all_text_lower):
        if "Midjourney" not in platforms:
            platforms.append("Midjourney")
        evidence.append("Midjourney Job ID signature in metadata")

    # 8. Provenance-focused text (avoid false-positive matching inside prompt text for short tokens)
    provenance_fields = [
        merged_texts.get("Software", ""),
        merged_texts.get("Source", ""),
        merged_texts.get("Author", ""),
        merged_texts.get("Creator", ""),
        merged_texts.get("DigitalSourceType", ""),
        merged_texts.get("Credit", ""),
        merged_texts.get("Model", ""),
        exif_tags.get("Software", ""),
        exif_tags.get("Make", ""),
        exif_tags.get("Model", ""),
        exif_tags.get("Artist", ""),
        exif_tags.get("Copyright", ""),
        exif_tags.get("ImageDescription", ""),
        *xmp_packets,
    ]
    provenance_lower = "\n".join(provenance_fields).lower()

    for needle, label, desc, word_boundary in PROVIDER_METADATA_RULES:
        haystack = provenance_lower if word_boundary else all_text_lower
        if _matches_rule(haystack, needle, word_boundary):
            if desc not in evidence:
                evidence.append(desc)
            if not platforms:
                platforms.append(label)

    detected = len(evidence) > 0
    platform = platforms[0] if platforms else None

    return {
        "detected": detected,
        "confidence": "DETERMINISTIC" if detected else "NONE",
        "platform": platform,
        "metadata_found": {
            "camera_exif_present": camera_exif_present,
            "text_chunks": {k: v[:1000] for k, v in merged_texts.items()},
            "exif": exif_tags,
            "aigc_record": aigc_record,
            "xmp_packets_count": len(xmp_packets),
            "xmp_preview": xmp_packets[0][:1200] if xmp_packets else None,
        },
        "chunks": chunks[:50],
        "evidence": evidence,
    }
