import tensorflow as tf
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.api.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

# Enable mixed precision training
from keras.api.mixed_precision import set_global_policy
from keras.api.models import Sequential
from keras.api.optimizers import Adam
from keras.api.utils import image_dataset_from_directory
from matplotlib import pyplot as plt

set_global_policy("mixed_float16")


train_dir = "./asl_alphabet_train"

# List all physical devices of type 'GPU'
gpus = tf.config.list_physical_devices("GPU")

# Check if any GPUs are available
if gpus:
    try:
        # Enable memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# Preprocessing and augmentation function
def preprocess_image(image, label):
    image = tf.image.resize(image, [200, 200])
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_hue(image, 0.1)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = image / 255.0  # Normalize to [0, 1]
    return image, label


# Create training and validation datasets
train_ds = (
    image_dataset_from_directory(
        train_dir,
        image_size=(200, 200),
        batch_size=32,  # Experiment with larger batch size
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="training",
    )
    .map(preprocess_image)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_ds = (
    image_dataset_from_directory(
        train_dir,
        image_size=(200, 200),
        batch_size=32,  # Experiment with larger batch size
        shuffle=True,
        seed=42,
        validation_split=0.2,
        subset="validation",
    )
    .map(lambda x, y: (x / 255.0, y))
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Define the number of classes
num_classes = 29


# One-hot encoding function
def one_hot_encode(image, label):
    label = tf.one_hot(label, depth=num_classes)
    return image, label


# Map the one-hot encoding function to the datasets
train_ds = train_ds.map(one_hot_encode)
val_ds = val_ds.map(one_hot_encode)

# Define the model architecture
model = Sequential(
    [
        Input(shape=(200, 200, 3)),
        Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(filters=256, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation="relu"),
        Dropout(0.4),
        BatchNormalization(),
        Dense(
            num_classes, activation="softmax", dtype="float32"
        ),  # Ensure float32 output
    ]
)

# Model summary
model.summary()

# Compile the model with mixed precision enabled
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)

callbacks = [early_stopping, reduce_lr]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,  # Increase if needed
    callbacks=callbacks,
)

# Save the model
model.save("asl_model_max_accuracy.h5")

# Plotting accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim([0, max(history.history["loss"])])
plt.legend(loc="upper right")

plt.show()
