"""CLI tool for unlimited batch non-ML AI watermark inspection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from watermark_engine.analyzer import analyze_image_file
from watermark_engine.sample_generator import generate_demo_samples

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"}


def _collect_targets(target: Path) -> list[Path]:
    """Collect single image file or all supported image files recursively."""
    if target.is_file():
        return [target]
    if target.is_dir():
        files = [
            p
            for p in sorted(target.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        ]
        return files
    return []


def _print_table(results: list[dict[str, Any]]) -> None:
    """Print an aligned ASCII/UTF-8 summary table of inspection results."""
    if not results:
        print("No supported image files found.")
        return

    headers = ("Filename", "Verdict", "Source / Carrier", "Triggered Layers")
    rows = [
        (
            r.get("filename", "")[:32],
            r.get("verdict", ""),
            (r.get("platform") or "-")[:34],
            ", ".join(r.get("triggered_layers") or ["-"]),
        )
        for r in results
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)

    print()
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()

    confirmed = sum(1 for r in results if r.get("verdict") == "WATERMARK_CONFIRMED")
    carrier = sum(1 for r in results if r.get("verdict") == "WATERMARK_CARRIER_DETECTED")
    clean = sum(1 for r in results if r.get("verdict") == "NO_WATERMARK_FOUND")
    print(
        f"Summary: Total={len(results)} | Confirmed (Layers 1-3)={confirmed} | "
        f"DSP Signal (Layer 4)={carrier} | None Found={clean}"
    )


def _export_csv(results: list[dict[str, Any]], csv_path: Path) -> None:
    """Write batch watermark scan results to CSV."""
    fieldnames = [
        "filename",
        "verdict",
        "platform",
        "triggered_layers",
        "format",
        "width",
        "height",
        "file_size_bytes",
        "sha256",
        "evidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            info = r.get("file_info", {})
            writer.writerow(
                {
                    "filename": r.get("filename", ""),
                    "verdict": r.get("verdict", ""),
                    "platform": r.get("platform") or "",
                    "triggered_layers": " | ".join(r.get("triggered_layers") or []),
                    "format": info.get("format", ""),
                    "width": info.get("width", ""),
                    "height": info.get("height", ""),
                    "file_size_bytes": info.get("file_size_bytes", ""),
                    "sha256": info.get("sha256", ""),
                    "evidence": " ; ".join(r.get("all_evidence") or []),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Non-ML Multi-Layer AI Watermark Inspector (Deterministic & DSP)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen_cmd = sub.add_parser("generate-samples", help="Generate 8 test images covering all 4 detection layers")
    gen_cmd.add_argument("--out", default="samples", help="Output directory for generated samples (default: samples)")

    scan_cmd = sub.add_parser("scan", help="Scan a single image or directory recursively without batch limits")
    scan_cmd.add_argument("target", help="Path to an image file or directory")
    scan_cmd.add_argument("--json", dest="json_out", help="Write full inspection report to JSON")
    scan_cmd.add_argument("--csv", dest="csv_out", help="Write summary table to CSV")

    args = parser.parse_args()

    if args.command == "generate-samples":
        created = generate_demo_samples(args.out)
        print(f"Generated {len(created)} demo samples in '{args.out}':")
        for item in created:
            print(f"  - {item['file']} -> {item['layer']}")
        return 0

    if args.command == "scan":
        target_path = Path(args.target)
        files = _collect_targets(target_path)
        if not files:
            print(f"Error: No supported image files found at '{target_path}'.", file=sys.stderr)
            return 1

        results = [analyze_image_file(f, include_visuals=False) for f in files]
        _print_table(results)

        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"Saved JSON report to: {args.json_out}")
        if args.csv_out:
            _export_csv(results, Path(args.csv_out))
            print(f"Saved CSV report to: {args.csv_out}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
