import os

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

path_to_plants = "dataset"
plant_classes = os.listdir(path_to_plants)
train_dir = 'dataset/plants'

# Create data generators
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
)

# Define batch size and image dimensions
batch_size = 50
img_height = 150
img_width = 150

# Load and prepare training data
train_data_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical for multi-class classification shuffle=True
    shuffle=True
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(4, activation='softmax')  # Update the number of classes to 4
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 10  # You can adjust the number of epochs as needed

history = model.fit(
    train_data_gen,
    epochs=epochs
)

# calculate accuracy for training data
train_loss, train_accuracy = model.evaluate(train_data_gen)

# Print the training accuracy
print("Training Accuracy:", train_accuracy)

# Get the class labels and their corresponding indices
class_labels = list(train_data_gen.class_indices.keys())

# Print the class labels
print("Class Labels:")
for label in class_labels:
    print(label)

model.save("models/classifier_sequential.h5")

if __name__ == '__main__':
    classes = input("Enter the image classes (seperated by spaces) that you'd like to classify: ").split(" ")
    for class : classes:

