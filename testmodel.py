import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Load the AI model
model = tf.keras.models.load_model("asl_model_max_accuracy.h5")

# Define the label dictionary
label_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}


# Function to load and preprocess the image
def load_image(image_path):
    img = Image.open(image_path).resize((200, 200))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


# Function to predict the label of the image
def predict_label(img_array):
    prediction = model.predict(img_array)
    label_index = np.argmax(prediction, axis=-1)[0]
    return label_dict[label_index]


# Function to open an image file and display it with its label
def open_image():
    file_path = filedialog.askopenfilename(
        initialdir="asl_alphabet_test",
        title="Select an Image",
        filetypes=(("jpeg files", "*.jpg"), ("all files", "*.*")),
    )
    if file_path:
        img, img_array = load_image(file_path)
        # resize the image to 200x200 pixels
        img = img.resize((200, 200))
        label = predict_label(img_array)

        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk
        label_var.set(f"Predicted Label: {label}")


# Create the main window
root = tk.Tk()
root.title("ASL Alphabet Recognition")

# Set the window size to 1K resolution (1024x768)
root.geometry("1024x768")

# Increase the scale of the UI
panel = tk.Label(root)
panel.pack(pady=40)

# Create a label to display the predicted label
label_var = tk.StringVar()
label_var.set("Predicted Label: ")
label_label = tk.Label(root, textvariable=label_var, font=("Helvetica", 32))
label_label.pack(pady=40)

# Create a button to open an image file
open_button = tk.Button(
    root,
    text="Open Image",
    command=open_image,
    font=("Helvetica", 24),
    width=20,
    height=2,
)
open_button.pack(pady=40)

# Start the Tkinter event loop
root.mainloop()
