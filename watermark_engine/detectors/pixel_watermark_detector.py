"""Layer 2: Invisible Pixel & Frequency Watermark Decoders (DWT-DCT-SVD, DWT-DCT & Stealth LSB)."""

from __future__ import annotations

import gzip
import struct
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn

STEALTH_HEADERS: dict[bytes, tuple[str, bool]] = {
    b"stealth_pnginfo": ("alpha", False),
    b"stealth_pngcomp": ("alpha", True),
    b"stealth_rgbinfo": ("rgb", False),
    b"stealth_rgbcomp": ("rgb", True),
}

# Verified fixed binary payloads from Stability AI, Diffusers, and Black Forest Labs FLUX
FLUX_48BIT_BYTES = b"\x2a\xfe\x87\x9c\xf5\x2e"  # 0b001010101111111010000111100111001111010100101110
SDXL_48BIT_BYTES = b"\xb3\xec\x90\x7b\xb1\x9e"  # 0b101100111110110010010000011110111011000110011110

KNOWN_FREQUENCY_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"SDV2", "Stability AI Stable Diffusion v2", "SDV2 (32-bit ASCII)"),
    (b"StableDiffusionV1", "Stability AI Stable Diffusion v1", "StableDiffusionV1 (136-bit ASCII)"),
    (b"SDXL", "Stability AI SDXL", "SDXL (32-bit ASCII)"),
    (FLUX_48BIT_BYTES, "Black Forest Labs (FLUX)", "0x2AFE879CF52E (FLUX 48-bit)"),
    (SDXL_48BIT_BYTES, "Stability AI SDXL (Diffusers)", "0xB3EC907BB19E (SDXL 48-bit)"),
)


def _haar_dwt2(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform 1-level orthonormal 2D Haar discrete wavelet transform."""
    c00 = channel[0::2, 0::2]
    c01 = channel[0::2, 1::2]
    c10 = channel[1::2, 0::2]
    c11 = channel[1::2, 1::2]
    ll = 0.5 * (c00 + c01 + c10 + c11)
    lh = 0.5 * (c00 - c01 + c10 - c11)
    hl = 0.5 * (c00 + c01 - c10 - c11)
    hh = 0.5 * (c00 - c01 - c10 + c11)
    return ll, lh, hl, hh


def _haar_idwt2(
    ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray
) -> np.ndarray:
    """Perform 1-level inverse 2D Haar discrete wavelet transform."""
    h, w = ll.shape
    out = np.empty((h * 2, w * 2), dtype=np.float64)
    out[0::2, 0::2] = 0.5 * (ll + lh + hl + hh)
    out[0::2, 1::2] = 0.5 * (ll - lh + hl - hh)
    out[1::2, 0::2] = 0.5 * (ll + lh - hl - hh)
    out[1::2, 1::2] = 0.5 * (ll - lh - hl + hh)
    return out


def embed_dwtdctsvd(
    img_rgb: np.ndarray, payload_bytes: bytes = b"SDV2", scale: float = 36.0
) -> np.ndarray:
    """Embed a repeating bitstream into the U channel via Haar DWT + 4x4 DCT + SVD."""
    if img_rgb.ndim != 3 or img_rgb.shape[2] < 3:
        raise ValueError("Expected RGB image array of shape (H, W, 3)")

    bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    num_bits = len(bits)
    if num_bits == 0:
        raise ValueError("Payload cannot be empty")

    rgb = img_rgb[:, :, :3].copy()
    h, w = rgb.shape[:2]
    h_use = (h // 8) * 8
    w_use = (w // 8) * 8
    if h_use < 16 or w_use < 16:
        raise ValueError("Image too small for DWT-DCT-SVD watermark")

    yuv = cv2.cvtColor(rgb[:h_use, :w_use], cv2.COLOR_RGB2YUV).astype(np.float64)
    u_chan = yuv[:, :, 1]
    ll, lh, hl, hh = _haar_dwt2(u_chan)

    ll_h, ll_w = ll.shape
    bit_idx = 0
    for r in range(0, ll_h - 3, 4):
        for c in range(0, ll_w - 3, 4):
            block = ll[r : r + 4, c : c + 4]
            dct_block = dctn(block, type=2, norm="ortho")
            u_mat, s_vals, vt_mat = np.linalg.svd(dct_block, full_matrices=False)
            bit = int(bits[bit_idx % num_bits])
            s_vals[0] = (np.floor(s_vals[0] / scale) + 0.25 + 0.5 * bit) * scale
            rebuilt_dct = (u_mat * s_vals) @ vt_mat
            ll[r : r + 4, c : c + 4] = idctn(rebuilt_dct, type=2, norm="ortho")
            bit_idx += 1

    yuv[:, :, 1] = _haar_idwt2(ll, lh, hl, hh)
    yuv_u8 = np.clip(np.round(yuv), 0, 255).astype(np.uint8)
    rgb[:h_use, :w_use] = cv2.cvtColor(yuv_u8, cv2.COLOR_YUV2RGB)

    if img_rgb.shape[2] == 4:
        out = img_rgb.copy()
        out[:, :, :3] = rgb
        return out
    return rgb


def _extract_block_qim_samples(
    img_rgb: np.ndarray, mode: str = "svd", scale: float = 36.0
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw binary votes and normalized QIM phase values from 4x4 LL DCT blocks."""
    h, w = img_rgb.shape[:2]
    h_use = (h // 8) * 8
    w_use = (w // 8) * 8
    if h_use < 16 or w_use < 16:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float64)

    yuv = cv2.cvtColor(img_rgb[:h_use, :w_use, :3], cv2.COLOR_RGB2YUV).astype(np.float64)
    ll, _, _, _ = _haar_dwt2(yuv[:, :, 1])
    ll_h, ll_w = ll.shape

    blocks_bits: list[int] = []
    phases: list[float] = []
    for r in range(0, ll_h - 3, 4):
        for c in range(0, ll_w - 3, 4):
            block = ll[r : r + 4, c : c + 4]
            dct_block = dctn(block, type=2, norm="ortho")
            if mode == "svd":
                coef = float(np.linalg.svd(dct_block, compute_uv=False)[0])
            else:
                coef = float(dct_block[0, 0])
            phase = (coef % scale) / scale
            phases.append(phase)
            blocks_bits.append(1 if phase > 0.5 else 0)

    return np.array(blocks_bits, dtype=np.uint8), np.array(phases, dtype=np.float64)


def _vote_bitstream(
    raw_bits: np.ndarray, phases: np.ndarray, num_bits: int
) -> dict[str, Any]:
    """Vote across repeated blocks of length num_bits and match known signatures."""
    if len(raw_bits) < num_bits * 2:
        return {
            "matched": False,
            "payload_bytes": b"",
            "payload_text": None,
            "bit_consistency": 0.0,
            "phase_alignment": 0.0,
            "repetitions": 0,
            "known_platform": None,
        }

    usable = (len(raw_bits) // num_bits) * num_bits
    matrix = raw_bits[:usable].reshape(-1, num_bits)
    repetitions = matrix.shape[0]

    ones_ratio = matrix.mean(axis=0)
    voted_bits = (ones_ratio >= 0.5).astype(np.uint8)
    per_bit_consistency = np.maximum(ones_ratio, 1.0 - ones_ratio)
    mean_consistency = float(per_bit_consistency.mean())

    dist_to_centers = np.minimum(np.abs(phases[:usable] - 0.25), np.abs(phases[:usable] - 0.75))
    phase_alignment = float(np.mean(dist_to_centers <= 0.14))

    payload_bytes = np.packbits(voted_bits).tobytes()
    known_platform: str | None = None
    signature_label: str | None = None
    for sig_bytes, platform_label, display_label in KNOWN_FREQUENCY_SIGNATURES:
        if len(payload_bytes) == len(sig_bytes) and payload_bytes == sig_bytes:
            known_platform = platform_label
            signature_label = display_label
            break

    is_printable = all(32 <= b <= 126 for b in payload_bytes)
    has_bit_diversity = 0.15 <= float(voted_bits.mean()) <= 0.85

    matched = False
    if known_platform is not None and mean_consistency >= 0.75:
        matched = True
    elif (
        is_printable
        and has_bit_diversity
        and repetitions >= 4
        and mean_consistency >= 0.86
        and phase_alignment >= 0.70
    ):
        matched = True

    if signature_label is not None and payload_bytes in (FLUX_48BIT_BYTES, SDXL_48BIT_BYTES):
        payload_text: str | None = signature_label
    elif matched or is_printable:
        payload_text = payload_bytes.decode("ascii", errors="replace")
    else:
        payload_text = None

    return {
        "matched": matched,
        "payload_bytes": payload_bytes,
        "payload_text": payload_text,
        "bit_consistency": round(mean_consistency, 4),
        "phase_alignment": round(phase_alignment, 4),
        "repetitions": repetitions,
        "known_platform": known_platform,
    }


def decode_dwtdctsvd(
    img_rgb: np.ndarray, num_bits: int = 32, scale: float = 36.0
) -> dict[str, Any]:
    """Decode a repeating DWT-DCT-SVD bitstream for a given bit-length."""
    raw_bits, phases = _extract_block_qim_samples(img_rgb, mode="svd", scale=scale)
    return _vote_bitstream(raw_bits, phases, num_bits)


def decode_dwtdct(
    img_rgb: np.ndarray, num_bits: int = 48, scale: float = 36.0
) -> dict[str, Any]:
    """Decode a repeating DWT-DCT DC-coefficient bitstream for a given bit-length."""
    raw_bits, phases = _extract_block_qim_samples(img_rgb, mode="dct_dc", scale=scale)
    return _vote_bitstream(raw_bits, phases, num_bits)


def embed_stealth_pnginfo(
    img_rgba: np.ndarray, text: str, compressed: bool = False
) -> np.ndarray:
    """Embed stealth_pnginfo or stealth_pngcomp bitstream into Alpha channel LSB in (x, y) order."""
    if img_rgba.ndim != 3 or img_rgba.shape[2] != 4:
        raise ValueError("Expected RGBA image array of shape (H, W, 4)")

    magic = b"stealth_pngcomp" if compressed else b"stealth_pnginfo"
    raw_text = text.encode("utf-8")
    payload = gzip.compress(raw_text) if compressed else raw_text
    bit_len = len(payload) * 8

    full_bytes = magic + struct.pack(">I", bit_len) + payload
    bits = np.unpackbits(np.frombuffer(full_bytes, dtype=np.uint8))

    out = img_rgba.copy()
    h, w = out.shape[:2]
    if len(bits) > h * w:
        raise ValueError("Payload exceeds image Alpha LSB capacity")

    alpha_col_major = out[:, :, 3].T.reshape(-1)
    alpha_col_major[: len(bits)] = (alpha_col_major[: len(bits)] & 0xFE) | bits
    out[:, :, 3] = alpha_col_major.reshape(w, h).T
    return out


def _try_decode_stealth_stream(
    bits: np.ndarray, channel_mode: str, scan_order: str
) -> dict[str, Any] | None:
    """Check if a 1D bit stream starts with a valid stealth_pnginfo / stealth_rgbinfo header."""
    if len(bits) < 152:
        return None

    header_bytes = np.packbits(bits[:120]).tobytes()
    if header_bytes not in STEALTH_HEADERS:
        return None

    expected_mode, is_compressed = STEALTH_HEADERS[header_bytes]
    if expected_mode != channel_mode:
        return None

    len_bytes = np.packbits(bits[120:152]).tobytes()
    payload_bit_len = struct.unpack(">I", len_bytes)[0]
    if payload_bit_len <= 0 or payload_bit_len % 8 != 0 or 152 + payload_bit_len > len(bits):
        return None

    payload_bytes = np.packbits(bits[152 : 152 + payload_bit_len]).tobytes()
    try:
        if is_compressed:
            decoded_text = gzip.decompress(payload_bytes).decode("utf-8")
        else:
            decoded_text = payload_bytes.decode("utf-8")
    except Exception:
        return None

    return {
        "header": header_bytes.decode("ascii"),
        "channel_mode": channel_mode,
        "scan_order": scan_order,
        "compressed": is_compressed,
        "payload_bits": payload_bit_len,
        "decoded_text": decoded_text,
    }


def decode_stealth_lsb(img_pil: Image.Image) -> dict[str, Any]:
    """Decode NovelAI / SD WebUI Stealth PNGInfo LSB bitstreams from Alpha or RGB channels."""
    arr = np.array(img_pil)
    candidates: list[tuple[np.ndarray, str, str]] = []

    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha_lsb = arr[:, :, 3] & 1
        candidates.append((alpha_lsb.T.reshape(-1), "alpha", "column-major (x,y)"))
        candidates.append((alpha_lsb.reshape(-1), "alpha", "row-major (y,x)"))

    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb_lsb = arr[:, :, :3] & 1
        candidates.append((rgb_lsb.transpose(1, 0, 2).reshape(-1), "rgb", "column-major (x,y)"))
        candidates.append((rgb_lsb.reshape(-1), "rgb", "row-major (y,x)"))

    for bit_stream, mode, order in candidates:
        res = _try_decode_stealth_stream(bit_stream, mode, order)
        if res is not None:
            return {"detected": True, **res}

    return {"detected": False}


def _classify_stealth_platform(text_lower: str) -> str:
    """Identify the generator from decoded Stealth LSB payload text."""
    if "novelai" in text_lower or ('"uc"' in text_lower and '"steps"' in text_lower):
        return "NovelAI (Stealth LSB PNGInfo)"
    if "sui_image_params" in text_lower or "swarmui" in text_lower:
        return "SwarmUI (Stealth LSB PNGInfo)"
    if "fooocus" in text_lower:
        return "Fooocus (Stealth LSB PNGInfo)"
    if '"class_type"' in text_lower and '"inputs"' in text_lower:
        return "ComfyUI (Stealth LSB PNGInfo)"
    if "draw things" in text_lower:
        return "Draw Things (Stealth LSB PNGInfo)"
    return "Stealth LSB PNGInfo (SD WebUI / Forge)"


def detect_pixel_watermarks(img_pil: Image.Image) -> dict[str, Any]:
    """Run all Layer-2 deterministic pixel & frequency bitstream decoders."""
    methods_matched: list[str] = []
    evidence: list[str] = []
    decoded_payload: str | None = None
    platform: str | None = None
    best_consistency = 0.0

    # 1. Stealth LSB PNGInfo (NovelAI / SD WebUI Stealth / Fooocus / SwarmUI)
    stealth_res = decode_stealth_lsb(img_pil)
    if stealth_res.get("detected"):
        methods_matched.append(f"LSB {stealth_res['header']} ({stealth_res['scan_order']})")
        decoded_payload = stealth_res["decoded_text"]
        platform = _classify_stealth_platform(decoded_payload.lower())
        best_consistency = 1.0
        evidence.append(
            f"Decoded LSB bitstream: {stealth_res['header']} "
            f"({stealth_res['channel_mode']}, {stealth_res['scan_order']}, {stealth_res['payload_bits']} bits)"
        )

    # 2. DWT-DCT-SVD & DWT-DCT (Stability AI SD v1/v2/SDXL, Black Forest Labs FLUX)
    rgb_arr = np.array(img_pil.convert("RGB"))
    svd_bits, svd_phases = _extract_block_qim_samples(rgb_arr, mode="svd", scale=36.0)
    dct_bits, dct_phases = _extract_block_qim_samples(rgb_arr, mode="dct_dc", scale=36.0)

    best_freq_res: dict[str, Any] = {}
    for method_label, bits_arr, phases_arr in (
        ("DWT-DCT-SVD", svd_bits, svd_phases),
        ("DWT-DCT", dct_bits, dct_phases),
    ):
        matched_this_method = False
        for candidate_bits in (48, 32, 136):
            freq_res = _vote_bitstream(bits_arr, phases_arr, candidate_bits)
            if freq_res["bit_consistency"] > best_consistency:
                best_consistency = freq_res["bit_consistency"]
                best_freq_res = freq_res
            if freq_res["matched"]:
                methods_matched.append(f"{method_label} ({candidate_bits}-bit)")
                if decoded_payload is None:
                    decoded_payload = freq_res["payload_text"]
                if platform is None:
                    platform = freq_res["known_platform"] or f"{method_label} Frequency Watermark"
                best_consistency = freq_res["bit_consistency"]
                best_freq_res = freq_res
                evidence.append(
                    f"Decoded {method_label} payload: {freq_res['payload_text']} "
                    f"({freq_res['bit_consistency'] * 100:.1f}% block consistency, "
                    f"{freq_res['repetitions']} repetitions)"
                )
                matched_this_method = True
                break
        if matched_this_method:
            break

    detected = len(methods_matched) > 0
    return {
        "detected": detected,
        "confidence": "DETERMINISTIC" if detected else "NONE",
        "platform": platform,
        "methods_matched": methods_matched,
        "decoded_payload": decoded_payload,
        "bit_consistency": round(best_consistency, 4),
        "details": {
            "stealth_lsb": stealth_res,
            "frequency_qim": {
                k: v for k, v in best_freq_res.items() if k != "payload_bytes"
            }
            if best_freq_res
            else {},
        },
        "evidence": evidence,
    }
