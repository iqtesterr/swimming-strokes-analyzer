import os
import sys

# Use my 5080!
os.environ["KERAS_BACKEND"] = "torch"

import csv
from pathlib import Path

import keras
import keras_hub
import numpy as np

from video_loader import STROKES, extract_features, load_videos, randomize

LOG_FIELDS = [
    "epoch", "accuracy", "loss", "val_accuracy", "val_loss",
    *(f"{stroke}_recall" for stroke in STROKES),
    "hyperparameters",
]

# Model Architectures
def train_classifier_1():
    """
    1. epochs = 50, batch = 1
    2. epochs = 100, batch = 1
    3. epochs = 50, batch = 4
    """
    return keras.Sequential([
        keras.layers.Input(shape=(768,)),
        keras.layers.Dense(4, activation="softmax"),
    ])


def train_classifier_2():
    """
    1. epochs = 50, batch = 1
    2. epochs = 100, batch = 4
    3. epochs = 76, batch = 4
    """
    return keras.Sequential([
        keras.layers.Input(shape=(768,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(4, activation="softmax"),
    ])


def train_classifier_3():
    """
    1. epochs = 100, batch = 1
    2. epochs = 100, batch = 4
    3. epochs = 100, batch = 2
    """
    return keras.Sequential([
        keras.layers.Input(shape=(768,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(4, activation="softmax"),
    ])

def train_classifier_4():
    """
    1. epochs = 100, batch = 1
    2. epochs = 100, batch = 4
    """
    return keras.Sequential([
        keras.layers.Input(shape=(16, 768)),
        keras.layers.Conv1D(64, 4, activation="relu"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(4, activation="softmax"),
    ])

def train_classifier_5():
    """
    1. epochs = 100, batch = 1
    2. epochs = 100, batch = 3
    3. epochs = 100, batch = 2
    """
    return keras.Sequential([
        keras.layers.Input(shape=(16, 768)),
        keras.layers.Conv1D(64, kernel_size=4),
        keras.layers.LeakyReLU(),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(4, activation="softmax"),
    ])

def train_classifier_6():
    """
    1. epochs = 100, batch = 1
    """
    return keras.Sequential([
        keras.layers.Input(shape=(16, 768)),
        keras.layers.Conv1D(64, kernel_size=4),
        keras.layers.LeakyReLU(),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(4, activation="softmax"),
    ])

# Logs
def to_input_arrays(examples):
    """Separate (features, label) pairs into two arrays."""
    features = np.stack([feature for feature, label in examples])
    labels = np.array([label for feature, label in examples])
    return features, labels


def get_log_paths(architecture_id: str):
    """Return new paths to log files for a given architecture."""
    folder = Path(__file__).parent/"eval"/"machine"/architecture_id
    folder.mkdir(parents=True, exist_ok=True)
    used_numbers = [
        int(path.stem) for path in folder.iterdir()
        if path.stem.isdigit() and path.suffix in (".csv", ".keras")
    ]
    number = max(used_numbers, default=0) + 1
    return folder/f"{number}.csv", folder/f"{number}.keras"


def start_log(model_architecture: str, optimizer: str, loss: str, batch_size: int, epochs: int):
    """Create a CSV file and return its path, model path, and hyperparameters string."""
    csv_path, model_path = get_log_paths(model_architecture.__name__)
    hyperparameters = (
        f"optimizer={optimizer}, "
        f"loss={loss}, "
        f"batch_size={batch_size}, "
        f"epochs={epochs}"
    )
    with csv_path.open("x", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=LOG_FIELDS).writeheader()
    return csv_path, model_path, hyperparameters


def record_epoch(csv_path, hyperparameters, epoch, history, model,
                 x_val, y_val):
    """
    Record one epoch of training and validation accuracy, loss,
    recall for each stroke on validation, and hyperparameters.
    """
    predictions = model.predict(x_val, verbose=0).argmax(axis=1)
    accuracy = history.history.get("accuracy")
    val_accuracy = history.history.get("val_accuracy")
    row = {
        "epoch": epoch,
        "accuracy": accuracy[0],
        "loss": history.history["loss"][0],
        "val_accuracy": val_accuracy[0],
        "val_loss": history.history["val_loss"][0],
        "hyperparameters": hyperparameters,
    }
    for label, stroke in enumerate(STROKES):
        # a boolean array aligned with validation labels,
        # where True represents the actual label is in this class
        actual = y_val == label
        row[f"{stroke}_recall"] = (
            # the fraction of actual samples of this class that predicted correctly
            float(np.mean(predictions[actual] == label)) if actual.any() else ""
        )
    # per row per epoch
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=LOG_FIELDS).writerow(row)

# Training
def train(
    model_architecture,
    train_examples,
    validation_examples,
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    batch_size=1,
    epochs=50,
    log=False,
):
    """
    Train a model with specific architecture and already extracted video features.
    """
    keras.utils.set_random_seed(5863)
    model = model_architecture()
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )
    x_val, y_val = to_input_arrays(validation_examples)

    if log:
        csv_path, model_path, hyperparameters = start_log(
            model_architecture, optimizer, loss, batch_size, epochs
        )

    for epoch in range(1, epochs + 1):
        x_train, y_train = to_input_arrays(randomize(train_examples, epoch))
        print(f"Epoch {epoch}/{epochs}")
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=1,
            shuffle=False,
            validation_data=(x_val, y_val),
        )
        if log:
            record_epoch(csv_path, hyperparameters, epoch, history, model, x_val, y_val)

    if log:
        model.save(model_path)
        print(f"Saved {csv_path} and {model_path}")
    return model


def show_training_curve(csv_path):
    """Open the accuracy, loss, and recall curves from one experiment CSV."""
    import matplotlib.pyplot as plt

    # load each row as dictionary
    with Path(csv_path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    # x-axis
    epochs = [int(row["epoch"]) for row in rows]
    # 1 figure with 3 plots
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for key in ("accuracy", "val_accuracy"):
        axes[0].plot(epochs, [float(row[key]) for row in rows], label=key)
    for key in ("loss", "val_loss"):
        axes[1].plot(epochs, [float(row[key]) for row in rows], label=key)
    for stroke in STROKES:
        key = f"{stroke}_recall"
        axes[2].plot(epochs, [float(row[key]) for row in rows], label=stroke)

    for axis, title in zip(axes, ("Accuracy", "Loss", "Validation Recall")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.legend()
    axes[0].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    #show_training_curve("eval/machine/train_classifier_5/4.csv")
    #sys.exit(0)

    video_prism = keras_hub.models.VideoPrismBackbone.from_preset(
        "videoprism_public_v1_base"
    )
    video_prism.trainable = False
    train_datas = extract_features(video_prism, load_videos("train"), avg_axis=(2,))
    validation_datas = extract_features(video_prism, load_videos("validation"), avg_axis=(2,))

    train(
        train_classifier_5,
        train_datas,
        validation_datas,
        log=True,
        epochs=53,
        batch_size=1
    )
