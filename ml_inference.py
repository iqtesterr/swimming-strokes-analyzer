import os
# Use my 5080!
os.environ["KERAS_BACKEND"] = "torch"

import keras
import keras_hub

video_prism = keras_hub.models.VideoPrismBackbone.from_preset(
    "videoprism_public_v1_base"
)

video_prism.trainable = False

model = keras.Sequential([
    # (batch, frames, height, width, channels)
    keras.layers.Input(shape=(1, 16, 288, 288, 3)),
    video_prism,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# process our videos

model.fit()