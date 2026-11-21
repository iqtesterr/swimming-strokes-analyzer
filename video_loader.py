import random
from pathlib import Path

import cv2
import numpy as np


DATA = Path(__file__).parent / "data"
STROKES = ("backstroke", "breaststroke", "butterfly", "freestyle")


def load_videos(split):
    """
    Load all videos of every stroke from a folder.
    The folder structure: root/split/<strokes>.
    """
    videos = []

    for i in range(len(STROKES)):
        folder = DATA / split / STROKES[i]
        for path in sorted(folder.glob("*.mp4")):
            videos.append((path, i))
    return videos

def split_frames(path):
    """Split 16 evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(str(path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for i in range(16):
        # just for robust
        if total_frames != 16:
            frame_index = round(i * (total_frames - 1) / 15)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames

def process_frames(frames):
    """Process each frame to the correct format of 288x288."""
    processed = []

    for frame in frames:
        frame = cv2.resize(frame, (288, 288), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed.append(frame)
    # (16, 288, 288, 3)
    return np.asarray(processed, dtype=np.float32) / 255.0


def randomize(features, epoch=1):
    """
    Randomize the order of feature-label pairs with a fixed seed + epoch.
    To better compare model architectures and hyperparameters.
    """
    shuffled = features.copy()
    random.Random(f"CSC211-wang5863-{epoch}").shuffle(shuffled)
    return shuffled

def extract_features(video_prism, videos, avg_axis=(1, 2)):
    """Get one feature vector and one label for each video."""
    examples = []

    for path, label in videos:
        video = process_frames(split_frames(path))
        # equivalent to (1 video, 16, 288, 288, 3)
        batch = np.expand_dims(video, axis=0)
        output = video_prism.predict(batch, verbose=0)
        # equivalent to keras.layers.GlobalAveragePooling2D()
        features = output.mean(axis=avg_axis)[0]
        examples.append((features, label))

    return examples
