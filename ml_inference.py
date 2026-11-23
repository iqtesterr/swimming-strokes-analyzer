import os
os.environ["KERAS_BACKEND"] = "torch"
import csv
from pathlib import Path

import keras
import keras_hub

from video_loader import STROKES, extract_features, load_videos


def get_ml_eval(model, samples):
    with open('ml_eval.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        csv.DictWriter(file, fieldnames=("example_id", "prediction")).writeheader()
        for key, value in samples.items():
            scores = model.predict(value[None, ...], verbose=0)[0]
            prediction = STROKES[int(scores.argmax())]
            writer.writerow((key, prediction))

def predict(model, sample):
    scores = model.predict(sample[1][None, ...], verbose=0)[0]
    for stroke, score in zip(STROKES, scores):
        print(f'{stroke}: {score:.3f}')
    print(f'Prediction of {sample[0].name}: {STROKES[scores.argmax()]}')

if __name__ == '__main__':
    run_test_set = True
    root = Path(__file__).resolve().parent
    classifier_path = root/"classifier.keras"
    video_prism = keras_hub.models.VideoPrismBackbone.from_preset(
        "videoprism_public_v1_base"
    )
    classifier = keras.saving.load_model(classifier_path, compile=False)

    if run_test_set:
        test_samples = {}
        videos = load_videos("test")
        features = extract_features(video_prism, videos, avg_axis=(2,))
        for i in range(len(videos)):
            path = videos[i][0]
            feature = features[i][0]
            test_samples[path.stem] = feature
        get_ml_eval(classifier, test_samples)
    else:
        videos = sorted(root.glob('*.mp4'))
        if len(videos) != 1:
            raise ValueError(f'Expected only one test video in {root}; found {len(videos)}')
        feature = extract_features(video_prism, [(videos[0], 0)], avg_axis=(2,))[0][0]
        predict(classifier, (videos[0], feature))
