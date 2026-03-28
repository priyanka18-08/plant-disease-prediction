import numpy as np
import os
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Path to dataset
data_path = "dataset/PlantVillage"

categories = os.listdir(data_path)

data = []
labels = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    label = categories.index(category)

    print("Loading:", category)

    for img in os.listdir(folder_path)[:500]:
        img_path = os.path.join(folder_path, img)

        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (64, 64))
            data.append(image)
            labels.append(label)
        except:
            pass
data = np.array(data)
labels = np.array(labels)

print("Total images:", len(data))
print("Total categories:", len(categories))
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
X_train = X_train / 255.0
X_test = X_test / 255.0
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(len(categories), activation='softmax'))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(X_train, y_train, epochs=5, batch_size=16)
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
model.save("plant_model.h5")
# --------- ADD HERE (prediction code) ---------

choice = input("Do you want to test image? (yes/no): ")

if choice == "yes":
    img_path = "test.jpg"

    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = np.array(img) / 255.0
    img = img.reshape(1, 64, 64, 3)

    prediction = model.predict(img)
    predicted_class = categories[np.argmax(prediction)]

    print("Predicted Disease:", predicted_class)