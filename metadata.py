"""Read image metadata without changing the original file."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


def extract_metadata(image_path: Path) -> dict[str, Any]:
    """Use ExifTool when installed; otherwise return Pillow EXIF fields."""
    if shutil.which("exiftool"):
        result = subprocess.run(
            ["exiftool", "-j", "-n", str(image_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode == 0:
            records = json.loads(result.stdout)
            return {"reader": "exiftool", "fields": records[0] if records else {}}

    with Image.open(image_path) as image:
        exifdata = image.getexif()
        fields = {
            ExifTags.TAGS.get(tag, str(tag)): value
            for tag, value in exifdata.items()
        }
    return {"reader": "pillow fallback", "fields": fields}


def metadata_flags(fields: dict[str, Any]) -> list[str]:
    """Return cautious review flags."""
    flags: list[str] = []
    date_original = fields.get("DateTimeOriginal")
    modified_date = fields.get("DateTime")
    create_date = fields.get("CreateDate") or fields.get("FileCreateDate")
    software = fields.get("Software") or fields.get("CreatorTool")

    if not date_original:
        flags.append("Original capture time is unavailable.")
    if modified_date:
        flags.append("Image has been modified") 
    if software:
        flags.append(f"Software metadata present: {software}")
    if date_original and create_date and str(date_original) > str(create_date):
        flags.append("Capture time is later than the recorded creation time.")
    if not fields.get("GPSLatitude"):
        flags.append("GPS location is unavailable.")
    return flags
