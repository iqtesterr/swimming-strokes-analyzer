"""Classify swimming videos using hand-selected binary mask templates.

Run this file to classify all data/test/<stroke>/*.mp4 files, or one
<root>/<example_id>.mp4 file when only a submitted sample is present.
"""
import csv
from pathlib import Path
from typing import cast

import cv2
import numpy as np

STROKES = ('backstroke', 'breaststroke', 'butterfly', 'freestyle')
# Geometric algorithm to handle noise
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Precompute the number of 1s for all 256 possible 8-pixel black-and-white patterns.
POPCOUNT = np.array([number.bit_count() for number in range(256)], dtype=np.uint8)


def skin_mask(frame, cr_min=120, cb_max=135):
    """Find skin-colored pixels and remove small spots."""
    # A common type of channel to match skin colors
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Based on the skin color to get the mask
    mask = cv2.inRange(ycrcb, (0, cr_min, 77), (255, 173, cb_max))
    # Noise reduction
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    # Fill the hole
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)


def swimmer_box(mask, previous):
    """Choose the swimmer's connected region, tracking its position for further mask cropping."""
    height, width = mask.shape
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates = []
    # Filter the regions that are too small (noise)
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area >= 150:
            candidates.append((x, y, w, h, area, centres[i]))
    if not candidates:
        return None
    # Use previous frame to judge if current box reasonable
    if previous is not None:
        px, py, pw, ph = previous
        near = [c for c in candidates
                if np.hypot(c[5][0]-px-pw/2, c[5][1]-py-ph/2)
                <= max(40, max(pw, ph)*.8)]
        if near:
            largest = max(c[4] for c in near)
            chosen = min(near, key=lambda c:
                         np.hypot(c[5][0]-px-pw/2, c[5][1]-py-ph/2)
                         / max(width, height)-.2*c[4]/largest)
            return chosen[:4]
    bodies = [c for c in candidates if c[5][1] > .4*height
              and c[3] > max(10, .025*height) and c[2] > 1.3*c[3]]
    return max(bodies or candidates, key=lambda c: c[4])[:4]


def video_masks(path):
    """Read a video and convert each frame into a 256x128 swimmer mask."""
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise FileNotFoundError(f'Cannot open video: {path}')
    masks = []
    previous = None
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break
            frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            mask = skin_mask(frame)
            box = swimmer_box(mask, previous)
            # Dark underwater frames may need a wider skin-color range
            if cv2.countNonZero(mask) < 150 or box is None:
                mask = skin_mask(frame, cr_min=110, cb_max=145)
                if box is None:
                    box = swimmer_box(mask, previous)
            # Keep the frame in the sequence even if no swimmer was found
            if box is None:
                masks.append(np.zeros((128, 256), np.uint8))
                previous = None
                continue
            x, y, w, h = box
            margin = round(.1*max(w, h))
            left, top = max(0, x-margin), max(0, y-margin)
            right, bottom = min(320, x+w+margin), min(180, y+h+margin)
            masks.append(cv2.resize(mask[top:bottom, left:right], (256, 128),
                                    interpolation=cv2.INTER_NEAREST))
            previous = box
    finally:
        video.release()
    if not masks:
        raise ValueError(f'No frames found: {path}')
    return np.stack(masks)


def load_templates(path):
    """Group template PNGs by stroke and pack their white pixels for matching."""
    images = []
    indices = {}
    for stroke in STROKES:
        start = len(images)
        for file in sorted((path/stroke).glob('*/*.png')):
            images.append(cast(np.ndarray, cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)))
        indices[stroke] = np.arange(start, len(images))
    masks = np.stack(images)
    # One bit per pixel makes comparison with all 1280 templates faster
    packed = np.packbits(masks.reshape(len(masks), -1) > 0, axis=1)
    areas = (masks > 0).sum(axis=(1, 2))
    return packed, areas, indices


def similarities(masks, templates):
    """Compare each video frame with every template using white-pixel IoU."""
    packed, areas, _ = templates
    scores = np.zeros((len(masks), len(packed)), np.float32)
    for i, mask in enumerate(masks):
        bits = np.packbits(mask > 0)
        overlap = POPCOUNT[np.bitwise_and(packed, bits)].sum(axis=1)
        union = areas + cv2.countNonZero(mask) - overlap
        # Black background does not count as a match
        scores[i] = overlap / np.maximum(union, 1)
    return scores


def predict(video_path, templates):
    """Average each class's five closest templates across all frames."""
    masks = video_masks(video_path)
    scores = {}
    for direction in (masks, masks[:, :, ::-1]):
        comparisons = similarities(direction, templates)
        for stroke, ids in templates[2].items():
            closest = np.partition(comparisons[:, ids], -5, axis=1)[:, -5:]
            scores[stroke] = max(float(closest.mean()), scores.get(stroke, 0))
    label = max(scores, key=lambda stroke: scores[stroke]) if max(scores.values()) > 0 else 'unknown'
    return label, scores


def write_predictions(folder, output, templates):
    """Predict every video in the different strokes folders."""
    rows = []
    seen = set()
    for stroke in STROKES:
        class_folder = folder/stroke
        if not class_folder.is_dir():
            raise FileNotFoundError(f'Missing class folder: {class_folder}')
        for video in sorted(class_folder.glob('*.mp4')):
            example_id = video.stem
            if example_id in seen:
                raise ValueError(f'Duplicate example ID: {example_id}')
            seen.add(example_id)
            rows.append((example_id, predict(video, templates)[0]))
    if not rows:
        raise ValueError(f'No videos found in {folder}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('example_id', 'prediction'))
        writer.writerows(rows)
    print(f'Wrote {len(rows)} predictions to {output}')


def write_validation_csv(base, templates):
    write_predictions(base/'data/validation', base/'eval/symbolic_validation.csv', templates)


def main():
    root = Path(__file__).resolve().parent
    templates = load_templates(root/'templates')
    test_folder = root/'data/test'
    # Evaluate entire test set or single test sample
    for stroke in STROKES:
        if (test_folder/stroke).is_dir():
            write_predictions(test_folder, root/'symbolic_eval.csv', templates)
            break
    else:
        videos = sorted(root.glob('*.mp4'))
        if len(videos) != 1:
            raise ValueError(f'Expected only one test video in {root}; found {len(videos)}')
        label, scores = predict(videos[0], templates)
        print(f'{videos[0].stem}: {label}')
        for stroke in STROKES:
            print(f'{stroke}: {scores[stroke]:.4f}')


if __name__ == '__main__':
    main()
