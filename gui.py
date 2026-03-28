import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import os

# Load trained model
model = load_model("plant_model.h5")

# Categories (same as training)
data_path = "dataset/PlantVillage"
categories = os.listdir(data_path)

# Create window
root = tk.Tk()
root.title("Plant Disease Detection")
root.geometry("400x500")

# Label
label = tk.Label(root, text="Upload Leaf Image", font=("Arial", 14))
label.pack(pady=10)

# Image display
img_label = tk.Label(root)
img_label.pack()

# Result label
result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=10)

# Function to upload and predict
def upload_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        # Show image
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        # Process image for model
        img_cv = cv2.imread(file_path)
        img_cv = cv2.resize(img_cv, (64, 64))
        img_cv = np.array(img_cv) / 255.0
        img_cv = img_cv.reshape(1, 64, 64, 3)

        # Predict
        prediction = model.predict(img_cv)
        predicted_class = categories[np.argmax(prediction)]

        result_label.config(text="Prediction: " + predicted_class)

# Button
btn = tk.Button(root, text="Upload Image", command=upload_image)
btn.pack(pady=20)

# Run app
root.mainloop()