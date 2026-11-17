"""Layer 1: Deterministic Cryptographic Provenance (C2PA / Content Credentials / JUMBF) scanner."""

from __future__ import annotations

import re
import struct
from typing import Any

# C2PA JUMBF Manifest Store UUID: d8fec3d6-1b0e-483c-9297-5828877ec481
C2PA_UUID = b"\xd8\xfe\xc3\xd6\x1b\x0e\x48\x3c\x92\x97\x58\x28\x87\x7e\xc4\x81"

JUMBF_BOX_MARKERS = (b"jumb", b"c2pa", b"c2ma", b"c2cl", b"c2cs", b"caBX")

# Ordered from most specific provider strings to broader vendor identifiers.
AI_GENERATOR_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("dall-e", "OpenAI DALL-E"),
    ("chatgpt", "OpenAI ChatGPT / GPT-4o"),
    ("sora", "OpenAI Sora"),
    ("openai", "OpenAI"),
    ("black forest labs", "Black Forest Labs (FLUX)"),
    ("bfl", "Black Forest Labs (FLUX)"),
    ("flux", "Black Forest Labs (FLUX)"),
    ("adobe firefly", "Adobe Firefly"),
    ("firefly", "Adobe Firefly"),
    ("adobe_stock", "Adobe Stock AI"),
    ("adobe_c2pa", "Adobe Content Credentials"),
    ("photoshop", "Adobe Photoshop Generative AI"),
    ("bing image creator", "Microsoft Bing Image Creator"),
    ("microsoft designer", "Microsoft Designer"),
    ("copilot", "Microsoft Copilot"),
    ("microsoft", "Microsoft AI"),
    ("synthid", "Google SynthID (C2PA)"),
    ("imagen", "Google Imagen"),
    ("gemini", "Google Gemini"),
    ("veo", "Google Veo"),
    ("made with google ai", "Google AI"),
    ("google", "Google AI"),
    ("magic media", "Canva Magic Media"),
    ("canva", "Canva AI"),
    ("ideogram", "Ideogram"),
    ("recraft", "Recraft AI"),
    ("leonardo.ai", "Leonardo AI"),
    ("leonardo ai", "Leonardo AI"),
    ("leonardo", "Leonardo AI"),
    ("imagine with meta", "Meta AI"),
    ("meta ai", "Meta AI"),
    ("meta platforms", "Meta AI"),
    ("midjourney", "Midjourney"),
    ("stability ai", "Stability AI"),
    ("stable diffusion", "Stability AI / Stable Diffusion"),
    ("sdxl", "Stability AI SDXL"),
    ("runwayml", "Runway AI"),
    ("runway", "Runway AI"),
    ("luma labs", "Luma Labs / Dream Machine"),
    ("dream machine", "Luma Labs / Dream Machine"),
    ("grok", "xAI Grok / Aurora"),
    ("aurora", "xAI Grok Aurora"),
    ("xai", "xAI Grok"),
    ("image playground", "Apple Image Playground"),
    ("apple intelligence", "Apple Intelligence"),
    ("galaxy ai", "Samsung Galaxy AI"),
    ("samsung", "Samsung Galaxy AI"),
    ("seedream", "ByteDance Seedream"),
    ("jimeng", "ByteDance Jimeng"),
    ("doubao", "ByteDance Doubao"),
    ("bytedance", "ByteDance AI"),
    ("tongyi", "Alibaba Tongyi Wanxiang"),
    ("wanx", "Alibaba Wanx"),
    ("qwen", "Alibaba Qwen"),
    ("kling", "Kuaishou Kling"),
    ("kolors", "Kuaishou Kolors"),
    ("kuaishou", "Kuaishou AI"),
    ("hunyuan", "Tencent Hunyuan"),
    ("tencent", "Tencent AI"),
    ("hailuo", "MiniMax Hailuo"),
    ("minimax", "MiniMax AI"),
    ("pikaso", "Freepik Pikaso"),
    ("freepik", "Freepik AI"),
)

DIGITAL_SOURCE_TYPES: tuple[str, ...] = (
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "algorithmicMedia",
)

C2PA_ACTIONS: tuple[str, ...] = (
    "c2pa.created",
    "c2pa.placed",
    "c2pa.edited",
    "c2pa.watermarked",
    "c2pa.converted",
)

_PRINTABLE_RE = re.compile(rb"[\x20-\x7e]{4,}")


def _extract_jpeg_app11_payloads(raw_bytes: bytes) -> list[bytes]:
    """Extract concatenated APP11 (0xFF 0xEB) JUMBF segments from a JPEG stream."""
    if not raw_bytes.startswith(b"\xff\xd8"):
        return []
    payloads: list[bytes] = []
    pos = 2
    length = len(raw_bytes)
    while pos + 4 <= length:
        if raw_bytes[pos] != 0xFF:
            break
        marker = raw_bytes[pos + 1]
        if marker in (0xD8, 0xD9):
            pos += 2
            continue
        if marker == 0xDA:  # Start of Scan (compressed image stream follows)
            break
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2
            continue
        seg_len = struct.unpack(">H", raw_bytes[pos + 2 : pos + 4])[0]
        if seg_len < 2 or pos + 2 + seg_len > length:
            break
        if marker == 0xEB:  # APP11 JUMBF
            payloads.append(raw_bytes[pos + 4 : pos + 2 + seg_len])
        pos += 2 + seg_len
    return payloads


def _extract_png_cabx_payloads(raw_bytes: bytes) -> list[bytes]:
    """Extract caBX (C2PA JUMBF) chunks from a PNG stream."""
    if not raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return []
    payloads: list[bytes] = []
    pos = 8
    length = len(raw_bytes)
    while pos + 12 <= length:
        chunk_len = struct.unpack(">I", raw_bytes[pos : pos + 4])[0]
        chunk_type = raw_bytes[pos + 4 : pos + 8]
        data_start = pos + 8
        data_end = data_start + chunk_len
        if data_end + 4 > length:
            break
        if chunk_type == b"caBX":
            payloads.append(raw_bytes[data_start:data_end])
        if chunk_type == b"IEND":
            break
        pos = data_end + 4
    return payloads


def _extract_webp_c2pa_payloads(raw_bytes: bytes) -> list[bytes]:
    """Extract C2PA or JUMB RIFF chunks from a WebP container."""
    if len(raw_bytes) < 12 or raw_bytes[:4] != b"RIFF" or raw_bytes[8:12] != b"WEBP":
        return []
    payloads: list[bytes] = []
    pos = 12
    length = len(raw_bytes)
    while pos + 8 <= length:
        fourcc = raw_bytes[pos : pos + 4]
        chunk_size = struct.unpack("<I", raw_bytes[pos + 4 : pos + 8])[0]
        data_start = pos + 8
        data_end = min(length, data_start + chunk_size)
        if fourcc in (b"C2PA", b"JUMB", b"caBX"):
            payloads.append(raw_bytes[data_start:data_end])
        pos = data_start + chunk_size + (chunk_size % 2)
    return payloads


def _scan_jumbf_boxes(blob: bytes) -> dict[str, Any]:
    """Scan binary blob for JUMBF superboxes, C2PA UUID, and printable manifest strings."""
    found_boxes: list[str] = []
    for box_tag in JUMBF_BOX_MARKERS:
        if box_tag in blob:
            found_boxes.append(box_tag.decode("ascii", errors="ignore"))

    has_c2pa_uuid = C2PA_UUID in blob
    strings = [
        m.decode("utf-8", errors="ignore").strip()
        for m in _PRINTABLE_RE.findall(blob)
    ]
    joined_lower = "\n".join(strings).lower()

    matched_platform: str | None = None
    for needle, label in AI_GENERATOR_SIGNATURES:
        if needle in joined_lower:
            matched_platform = label
            break

    matched_sources = [dst for dst in DIGITAL_SOURCE_TYPES if dst.lower() in joined_lower]
    matched_actions = [act for act in C2PA_ACTIONS if act in joined_lower]

    claim_generator = None
    for idx, token in enumerate(strings):
        low = token.lower()
        if "claim_generator" in low or "softwareagent" in low:
            # Grab nearby descriptive token if the key itself does not carry the value
            if idx + 1 < len(strings) and len(strings[idx + 1]) >= 3:
                claim_generator = strings[idx + 1][:120]
            else:
                claim_generator = token[:120]
            break

    return {
        "has_c2pa_uuid": has_c2pa_uuid,
        "jumbf_boxes": found_boxes,
        "matched_platform": matched_platform,
        "digital_source_types": matched_sources,
        "actions": matched_actions,
        "claim_generator": claim_generator,
        "sample_strings": strings[:30],
    }


def detect_c2pa(raw_bytes: bytes, filename: str = "") -> dict[str, Any]:
    """Detect C2PA / Content Credentials / JUMBF provenance containers deterministically."""
    app11_payloads = _extract_jpeg_app11_payloads(raw_bytes)
    cabx_payloads = _extract_png_cabx_payloads(raw_bytes)
    webp_payloads = _extract_webp_c2pa_payloads(raw_bytes)

    container_blobs = app11_payloads + cabx_payloads + webp_payloads
    container_source = (
        "JPEG APP11 (JUMBF)"
        if app11_payloads
        else "PNG caBX"
        if cabx_payloads
        else "WebP RIFF C2PA"
        if webp_payloads
        else None
    )

    # Fallback raw binary scan for ISO BMFF / AVIF / MP4 or non-standard JUMBF offsets
    if not container_blobs and (C2PA_UUID in raw_bytes or (b"jumb" in raw_bytes and b"c2pa" in raw_bytes)):
        idx = raw_bytes.find(b"jumb")
        if idx == -1:
            idx = raw_bytes.find(C2PA_UUID)
        start = max(0, idx - 32)
        end = min(len(raw_bytes), idx + 65536)
        container_blobs = [raw_bytes[start:end]]
        container_source = "Raw JUMBF/C2PA Binary Box"

    if not container_blobs:
        return {
            "detected": False,
            "confidence": "NONE",
            "platform": None,
            "manifest_summary": {},
            "evidence": [],
        }

    combined_blob = b"\n".join(container_blobs)
    scan = _scan_jumbf_boxes(combined_blob)

    is_c2pa = bool(
        scan["has_c2pa_uuid"]
        or any(b in scan["jumbf_boxes"] for b in ("c2pa", "c2ma", "c2cl", "c2cs", "caBX"))
        or (app11_payloads and b"jumb" in combined_blob)
    )

    if not is_c2pa:
        return {
            "detected": False,
            "confidence": "NONE",
            "platform": None,
            "manifest_summary": {},
            "evidence": [],
        }

    evidence: list[str] = [f"C2PA/JUMBF segment: {container_source}"]
    if scan["has_c2pa_uuid"]:
        evidence.append("C2PA Manifest Store UUID verified (d8fec3d6-1b0e-483c-9297-5828877ec481)")
    if scan["jumbf_boxes"]:
        evidence.append(f"JUMBF boxes: {', '.join(scan['jumbf_boxes'])}")
    if scan["claim_generator"]:
        evidence.append(f"Claim Generator: {scan['claim_generator']}")
    if scan["digital_source_types"]:
        evidence.append(f"DigitalSourceType: {', '.join(scan['digital_source_types'])}")
    if scan["actions"]:
        evidence.append(f"C2PA actions: {', '.join(scan['actions'])}")

    platform = scan["matched_platform"] or (
        "C2PA AI Provenance (Trained Algorithmic Media)"
        if scan["digital_source_types"]
        else "C2PA Content Credentials"
    )

    return {
        "detected": True,
        "confidence": "DETERMINISTIC",
        "platform": platform,
        "manifest_summary": {
            "container_source": container_source,
            "c2pa_uuid_present": scan["has_c2pa_uuid"],
            "jumbf_boxes": scan["jumbf_boxes"],
            "claim_generator": scan["claim_generator"],
            "digital_source_types": scan["digital_source_types"],
            "actions": scan["actions"],
            "payload_bytes": len(combined_blob),
            "extracted_strings": scan["sample_strings"],
        },
        "evidence": evidence,
    }
