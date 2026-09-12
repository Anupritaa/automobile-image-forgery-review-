from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2

from .metadata import extract_metadata, metadata_flags
from .quality import assess_quality
from .copy_move import detect_copy_move


def analyze_image(image_path: Path) -> dict:
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    metadata_result = extract_metadata(image_path)
    quality_result = assess_quality(image)
    copy_move_result, annotated_image = detect_copy_move(image)

    safire_map = Path("results/safire") / f"{image_path.name}.png"

    report = {
        "image": str(image_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": {
            "fields": metadata_result["fields"],
            "review_flags": metadata_flags(metadata_result["fields"]),
        },
        "image_quality": quality_result,
        "copy_move": copy_move_result,
        "safire": {
            "map_available": safire_map.exists(),
            "map_path": str(safire_map) if safire_map.exists() else None,
            "note": "SAFIRE map is review evidence, not a final forgery decision.",
        },
    }

    output_dir = Path("results/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{image_path.stem}_report.json"
    overlay_path = output_dir / f"{image_path.stem}_copy_move.png"

    report_path.write_text(json.dumps(report, indent=2, default=str))
    cv2.imwrite(str(overlay_path), annotated_image)

    return {
        "report_path": report_path,
        "copy_move_overlay": overlay_path,
        "report": report,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a JPG or PNG image")
    args = parser.parse_args()

    result = analyze_image(Path(args.image))

    print(json.dumps(result["report"], indent=2, default=str))
    print(f"\nSaved report: {result['report_path']}")
    print(f"Saved copy-move overlay: {result['copy_move_overlay']}")