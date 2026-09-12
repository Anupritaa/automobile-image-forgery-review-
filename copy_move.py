"""Baseline ORB/RANSAC copy-move detector for review assistance."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def detect_copy_move(image: np.ndarray, min_spatial_distance: float = 60.0) -> tuple[dict[str, Any], np.ndarray]:
    """Find repeated feature patterns; output is evidence, never a final verdict."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2500)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    annotated = image.copy()

    if descriptors is None or len(keypoints) < 6:
        return ({"status": "limited", "reason": "Too few visual features for copy-move analysis.", "matches": 0}, annotated)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates = matcher.knnMatch(descriptors, descriptors, k=3)
    matches = []
    for query_index, neighbors in enumerate(candidates):
        usable = next((item for item in neighbors if item.trainIdx != query_index), None)
        if usable is None:
            continue
        source = np.array(keypoints[query_index].pt)
        target = np.array(keypoints[usable.trainIdx].pt)
        if np.linalg.norm(source - target) >= min_spatial_distance:
            matches.append(usable)

    if len(matches) < 4:
        return ({"status": "no strong evidence", "reason": "No geometrically testable repeated regions found.", "matches": len(matches)}, annotated)

    source_points = np.float32([keypoints[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
    target_points = np.float32([keypoints[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
    _, inlier_mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)
    inliers = np.flatnonzero(inlier_mask.ravel()) if inlier_mask is not None else np.array([], dtype=int)

    for index in inliers[:80]:
        match = matches[int(index)]
        source = tuple(map(int, keypoints[match.queryIdx].pt))
        target = tuple(map(int, keypoints[match.trainIdx].pt))
        cv2.circle(annotated, source, 8, (0, 165, 255), 2)
        cv2.circle(annotated, target, 8, (0, 0, 255), 2)
        cv2.line(annotated, source, target, (255, 0, 255), 1)

    status = "possible_copy_move" if len(inliers) >= 15 else "no_strong_evidence"
    return (
        {
            "status": status,
            "keypoints": len(keypoints),
            "candidate_matches": len(matches),
            "geometric_inliers": int(len(inliers)),
            "reason": "Repeated features with geometric consistency require human review." if status == "possible_copy_move" else "Too few geometrically consistent matches.",
        },
        annotated,
    )

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to a JPG or PNG image")
    parser.add_argument(
        "--output",
        default="copy_move_overlay.png",
        help="Where to save the annotated image",
    )
    args = parser.parse_args()

    image = cv2.imread(str(Path(args.image)))
    if image is None:
        parser.error("Could not read the image file.")

    result, annotated = detect_copy_move(image)

    cv2.imwrite(args.output, annotated)
    print(json.dumps(result, indent=2))
    print(f"Annotated image saved to: {args.output}")