import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import os
disease_info = {

    "Pepper__bell___Bacterial_spot": (
        "Bacterial disease causing leaf spots",
        "Use copper-based bactericide and remove infected leaves"
    ),

    "Pepper__bell___healthy": (
        "Healthy plant",
        "No treatment needed"
    ),

    "Potato___Early_blight": (
        "Fungal disease causing brown spots",
        "Use fungicide and avoid overwatering"
    ),

    "Potato___Late_blight": (
        "Serious fungal disease",
        "Remove infected plants and apply fungicide immediately"
    ),

    "Potato___healthy": (
        "Healthy plant",
        "No treatment needed"
    ),

    "Tomato_Bacterial_spot": (
        "Bacterial infection causing leaf spots",
        "Use copper spray and avoid overhead watering"
    ),

    "Tomato_Early_blight": (
        "Fungal disease causing concentric spots",
        "Remove infected leaves and spray fungicide"
    ),

    "Tomato_healthy": (
        "Healthy plant",
        "No treatment needed"
    ),

    "Tomato_Late_blight": (
        "Fast spreading fungal disease",
        "Use fungicide and remove infected leaves immediately"
    ),

    "Tomato_Leaf_Mold": (
        "Fungal disease due to humidity",
        "Improve air circulation and apply fungicide"
    ),

    "Tomato_Septoria_leaf_spot": (
        "Fungal disease causing small spots",
        "Remove infected leaves and use fungicide"
    ),

    "Tomato_Spider_mites_Two_spotted_spider_mite": (
        "Pest infestation causing leaf damage",
        "Use insecticidal soap or neem oil"
    ),

    "Tomato__Target_Spot": (
        "Fungal disease causing target-like spots",
        "Apply fungicide and remove infected parts"
    ),

    "Tomato__Tomato_mosaic_virus": (
        "Viral disease causing mosaic patterns",
        "Remove infected plants and disinfect tools"
    ),

    "Tomato__Tomato_YellowLeaf__Curl_Virus": (
        "Viral disease causing yellow curling leaves",
        "Control whiteflies and remove infected plants"
    )
}
# Load trained model
model = load_model("plant_model.h5")

# Categories (same as training)
data_path = "dataset/PlantVillage"
categories = os.listdir(data_path)

# Create window
root = tk.Tk()
root.title("Plant Disease Detection")
root.geometry("500x600")
root.configure(bg="#e6f2ff")   # 🎨 background color

# Label
label = tk.Label(root, text="🌿 Plant Disease Detection", font=("Arial", 16, "bold"), bg="#e6f2ff")
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

        # Process image
        img_cv = cv2.imread(file_path)
        img_cv = cv2.resize(img_cv, (64, 64))
        img_cv = np.array(img_cv) / 255.0
        img_cv = img_cv.reshape(1, 64, 64, 3)

        # Predict
        prediction = model.predict(img_cv)
        predicted_class = categories[np.argmax(prediction)]
        confidence = np.max(prediction) * 100


        info, solution = disease_info.get(
            predicted_class,
            ("Unknown", "No solution available")
        )

        result_text = f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%\n\nDisease Info: {info}\nTreatment: {solution}"

        # COLOR CODE
        if "healthy" in predicted_class.lower():
            bg_color = "#ccffcc"
            text_color = "green"
        else:
            bg_color = "#ffcccc"
            text_color = "red"

        result_label.config(text=result_text, bg=bg_color, fg=text_color)
# Button
btn = tk.Button(root, text="Upload Image", command=upload_image,
                bg="green", fg="white", font=("Arial", 12))
btn.pack(pady=20)

# Run app
root.mainloop()