"""Layer 4: Pure DSP Frequency-Domain & Wavelet Carrier Analysis (Zero ML)."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import maximum_filter, median_filter, uniform_filter1d
from scipy.signal.windows import tukey


def _encode_png_b64(bgr_or_gray: np.ndarray) -> str:
    """Encode an OpenCV image array as a base64 PNG data URI."""
    ok, buf = cv2.imencode(".png", bgr_or_gray)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _resize_for_preview(img: np.ndarray, max_dim: int = 320, interp: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize diagnostic visualization while preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _center_crop_max(arr: np.ndarray, max_size: int = 512) -> np.ndarray:
    """Center-crop to at most max_size x max_size without resampling pixel frequencies."""
    h, w = arr.shape[:2]
    ch = min(h, max_size)
    cw = min(w, max_size)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return arr[y0 : y0 + ch, x0 : x0 + cw]


def _haar_hh_kurtosis(gray: np.ndarray) -> float:
    """Compute excess kurtosis of the 2D Haar HH diagonal detail subband."""
    h = (gray.shape[0] // 2) * 2
    w = (gray.shape[1] // 2) * 2
    if h < 16 or w < 16:
        return 0.0
    g = gray[:h, :w]
    hh = 0.5 * (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2])
    std = float(np.std(hh))
    if std < 0.25:
        return 0.0
    centered = (hh - np.mean(hh)) / std
    return float(np.mean(centered**4) - 3.0)


def _compute_svd_phase_concentration(rgb_crop: np.ndarray, scale: float = 36.0) -> float:
    """Check 4x4 LL DCT singular-value QIM clustering concentration around {0.25, 0.75}."""
    h = (rgb_crop.shape[0] // 8) * 8
    w = (rgb_crop.shape[1] // 8) * 8
    if h < 32 or w < 32:
        return 0.0
    yuv = cv2.cvtColor(rgb_crop[:h, :w], cv2.COLOR_RGB2YUV).astype(np.float64)
    u_ch = yuv[:, :, 1]
    ll = 0.5 * (u_ch[0::2, 0::2] + u_ch[0::2, 1::2] + u_ch[1::2, 0::2] + u_ch[1::2, 1::2])
    phases: list[float] = []
    for r in range(0, ll.shape[0] - 3, 4):
        for c in range(0, ll.shape[1] - 3, 4):
            dct_blk = dctn(ll[r : r + 4, c : c + 4], type=2, norm="ortho")
            s0 = float(np.linalg.svd(dct_blk, compute_uv=False)[0])
            phases.append((s0 % scale) / scale)
    if not phases:
        return 0.0
    p_arr = np.array(phases)
    dist = np.minimum(np.abs(p_arr - 0.25), np.abs(p_arr - 0.75))
    return float(np.mean(dist <= 0.12))


def analyze_dsp_carriers(img_pil: Image.Image, include_visuals: bool = True) -> dict[str, Any]:
    """Perform pure DSP 2D FFT, radial ring, wavelet, and LSB diagnostic analysis."""
    rgb_full = np.array(img_pil.convert("RGB"))
    rgb_crop = _center_crop_max(rgb_full, max_size=512)
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float64)
    h, w = gray.shape

    # High-pass spatial residual removes image content while preserving embedded carriers
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    residual = gray - blurred

    # Smooth Tukey taper eliminates rectangular boundary cross-leakage in FFT
    win_y = tukey(h, alpha=0.12)
    win_x = tukey(w, alpha=0.12)
    windowed = residual * np.outer(win_y, win_x)

    fft_shift = np.fft.fftshift(np.fft.fft2(windowed))
    mag = np.log1p(np.abs(fft_shift))

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = min(cy, cx)

    # Mask DC center, horizontal/vertical axes (+-2 px), and extreme corners
    valid_mask = (
        (radius >= max(8.0, 0.10 * r_max))
        & (radius <= 0.94 * r_max)
        & (np.abs(yy - cy) > 2)
        & (np.abs(xx - cx) > 2)
    )

    valid_vals = mag[valid_mask]
    global_sigma = float(np.std(valid_vals)) if valid_vals.size > 0 else 1.0

    local_med = median_filter(mag, size=9)
    local_mad = median_filter(np.abs(mag - local_med), size=9)
    local_sigma = np.maximum(1.4826 * local_mad, global_sigma * 0.40)

    z_map = np.where(valid_mask, (mag - local_med) / (local_sigma + 1e-6), 0.0)
    is_local_max = (maximum_filter(z_map, size=5) == z_map) & (z_map > 5.8)

    peak_coords = np.argwhere(is_local_max)
    carrier_peak_count = int(len(peak_coords))
    max_peak_zscore = float(np.max(z_map)) if z_map.size > 0 else 0.0

    # Radial ring modulation analysis (Tree-Ring watermark detector)
    ring_r_min = max(10, int(0.10 * r_max))
    ring_r_max = max(ring_r_min + 12, int(0.75 * r_max))
    r_int = np.rint(radius).astype(np.int32)
    radial_profile = []
    for r_val in range(ring_r_min, ring_r_max):
        ring_mask = valid_mask & (r_int == r_val)
        if np.any(ring_mask):
            radial_profile.append(float(np.mean(mag[ring_mask])))

    ring_modulation_score = 0.0
    max_ring_delta = 0.0
    if len(radial_profile) >= 15:
        rp = np.array(radial_profile, dtype=np.float64)
        trend = uniform_filter1d(rp, size=9, mode="nearest")
        detrended = rp - trend
        max_ring_delta = float(np.max(detrended))
        ring_modulation_score = float(np.std(detrended) / (global_sigma + 1e-6))

    hh_kurtosis = _haar_hh_kurtosis(gray)
    svd_phase_conc = _compute_svd_phase_concentration(rgb_crop)

    # Pure deterministic DSP thresholds (no ML)
    has_lattice_peaks = carrier_peak_count >= 4 and max_peak_zscore >= 6.2
    has_strong_peak = carrier_peak_count >= 2 and max_peak_zscore >= 7.8
    has_ring_carrier = max_ring_delta >= 0.45 and ring_modulation_score >= 0.18

    detected = bool(has_lattice_peaks or has_strong_peak or has_ring_carrier)

    carrier_types: list[str] = []
    evidence: list[str] = []
    if has_lattice_peaks or has_strong_peak:
        carrier_types.append("2D FFT Periodic Carrier")
        evidence.append(
            f"Detected {carrier_peak_count} narrowband 2D FFT peaks (max Z-score: {max_peak_zscore:.2f})"
        )
    if has_ring_carrier:
        carrier_types.append("Concentric Ring Carrier")
        evidence.append(
            f"Detected radial ring modulation (delta: {max_ring_delta:.3f}, energy ratio: {ring_modulation_score:.3f})"
        )

    carrier_type = " + ".join(carrier_types) if carrier_types else None

    visuals: dict[str, str] = {}
    if include_visuals:
        # 1. 2D FFT Magnitude Spectrum + Highlighted Carrier Peaks
        lo = float(np.percentile(mag, 5))
        hi = float(np.percentile(mag, 99.7))
        norm_mag = np.clip((mag - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        fft_vis = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)
        fft_vis = _resize_for_preview(fft_vis, max_dim=320, interp=cv2.INTER_AREA)
        scale_y = fft_vis.shape[0] / float(h)
        scale_x = fft_vis.shape[1] / float(w)
        for py, px in peak_coords[:24]:
            cx_vis = int(round(px * scale_x))
            cy_vis = int(round(py * scale_y))
            cv2.circle(fft_vis, (cx_vis, cy_vis), 7, (255, 255, 0), 1, lineType=cv2.LINE_AA)
        visuals["fft_spectrum_b64"] = _encode_png_b64(fft_vis)

        # 2. LSB Bit-Plane Map (Alpha LSB if RGBA and non-constant, else Green LSB)
        raw_arr = np.array(img_pil)
        raw_crop = _center_crop_max(raw_arr, max_size=512)
        if raw_crop.ndim == 3 and raw_crop.shape[2] == 4:
            alpha_lsb = (raw_crop[:, :, 3] & 1) * 255
            if np.ptp(alpha_lsb) > 0:
                lsb_plane = alpha_lsb.astype(np.uint8)
            else:
                lsb_plane = ((raw_crop[:, :, 1] & 1) * 255).astype(np.uint8)
        elif raw_crop.ndim == 3 and raw_crop.shape[2] >= 2:
            lsb_plane = ((raw_crop[:, :, 1] & 1) * 255).astype(np.uint8)
        else:
            lsb_plane = ((raw_crop & 1) * 255).astype(np.uint8)
        lsb_vis = _resize_for_preview(lsb_plane, max_dim=320, interp=cv2.INTER_NEAREST)
        visuals["lsb_plane_b64"] = _encode_png_b64(lsb_vis)

        # 3. 20x Amplified High-Pass Spatial Residual Map
        res_amp = np.clip(np.abs(residual) * 20.0, 0, 255).astype(np.uint8)
        res_vis = cv2.applyColorMap(res_amp, cv2.COLORMAP_VIRIDIS)
        res_vis = _resize_for_preview(res_vis, max_dim=320, interp=cv2.INTER_AREA)
        visuals["residual_map_b64"] = _encode_png_b64(res_vis)

    return {
        "detected": detected,
        "confidence": "DSP_SIGNAL" if detected else "NONE",
        "carrier_type": carrier_type,
        "metrics": {
            "carrier_peak_count": carrier_peak_count,
            "max_peak_zscore": round(max_peak_zscore, 3),
            "ring_modulation_score": round(ring_modulation_score, 4),
            "max_ring_delta": round(max_ring_delta, 4),
            "hh_wavelet_kurtosis": round(hh_kurtosis, 3),
            "svd_phase_concentration": round(svd_phase_conc, 4),
            "analyzed_region": f"{w}x{h}",
        },
        "visuals": visuals,
        "evidence": evidence,
    }
