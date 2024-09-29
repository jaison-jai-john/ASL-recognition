import pandas as pd
import tensorflow as tf
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.api.utils import image_dataset_from_directory
from matplotlib import pyplot as plt

train_dir = "./asl_alphabet_train"


def preprocess_image(image, label):
    image = tf.image.resize(image, [200, 200])
    image = image / 255.0  # Normalize to [0, 1]
    return image, label


train_ds = (
    image_dataset_from_directory(
        train_dir,
        image_size=(200, 200),
        batch_size=128,  # Adjust batch size if necessary
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="training",
    )
    .map(preprocess_image)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

val_ds = (
    image_dataset_from_directory(
        train_dir,
        image_size=(200, 200),
        batch_size=128,  # Adjust batch size if necessary
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="validation",
    )
    .map(preprocess_image)  # Use preprocess_image to ensure consistency
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

# Define the number of classes
num_classes = 29


# Function to one-hot encode the labels
def one_hot_encode(image, label):
    label = tf.one_hot(label, depth=num_classes)
    return image, label


# Map the one-hot encoding function to the datasets
train_ds = train_ds.map(one_hot_encode)
val_ds = val_ds.map(one_hot_encode)

model = Sequential(
    [
        Conv2D(
            filters=32, kernel_size=(3, 3), activation="relu", input_shape=(200, 200, 3)
        ),
        MaxPooling2D(2, 2),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        BatchNormalization(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        BatchNormalization(),
        Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# Compile the model
model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# EarlyStopping callback: Stops training when 'val_loss' does not improve.
early_stopping = EarlyStopping(
    monitor="val_loss",  # Monitors validation loss
    patience=3,  # Number of epochs with no improvement before stopping
    restore_best_weights=True,  # Restore the weights from the epoch with the best 'val_loss'
)

# ReduceLROnPlateau callback: Reduces the learning rate when 'val_loss' plateaus.
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",  # Monitors validation loss
    factor=0.1,  # Factor by which the learning rate will be reduced
    patience=2,  # Number of epochs with no improvement before reducing the learning rate
    min_lr=1e-6,  # Lower bound on the learning rate
)

# Combine the callbacks into a list
callbacks = [early_stopping, reduce_lr]

# Train the model with the callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=4,  # Set a higher number of epochs
    callbacks=callbacks,
)

history_df = pd.DataFrame(history.history)

model.save("asl_model.h5")
