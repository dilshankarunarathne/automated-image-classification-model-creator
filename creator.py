import os

from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from image_miner import downloader

path_to_plants = "dataset"
plant_classes = os.listdir(path_to_plants)
train_dir = 'dataset/train'

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


def train(epochs=15):
    history = model.fit(
        train_data_gen,
        epochs=epochs
    )
    model.save("models/classifier_sequential.h5")


def evaluate():
    # calculate accuracy for training data
    train_loss, train_accuracy = model.evaluate(train_data_gen)

    # Print the training accuracy
    print("Training Accuracy:", train_accuracy)


if __name__ == '__main__':
    classes = input("Enter the image classes (seperated by spaces) that you'd like to classify: ").split(" ")
    num_images = int(input("Enter the number of images for a class in the dataset: "))

    # create train dataset
    print("train dataset seperation started")
    for (root, dirs, files) in os.walk('dataset', topdown=True):
        if root == 'train' or root == 'dataset':
            print("skipping ", root, " directory from train dataset move...")
            continue
        for file in files:


    # create dataset
    for cls in classes:
        downloader.download(
            cls,
            limit=num_images,
            output_dir='dataset',
            adult_filter_off=False,
            force_replace=False,
            timeout=60,
            verbose=True
        )

    epc = int(input("Enter the number of epochs: "))
    train(epc)
