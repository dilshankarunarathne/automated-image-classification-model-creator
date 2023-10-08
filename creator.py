import os
import shutil

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


def train(epochs=15):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(n_classes, activation='softmax')  # Update the number of classes to 4
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

    n_classes = len(classes)

    # create train dataset
    print("train dataset separation started")
    for cl in classes:
        folder = './dataset/' + cl
        for (root, dirs, files) in os.walk(folder, topdown=True):
            for file in files:
                print("working on ", root, " directory dataset")
                src_path = root + "/" + file
                dst_path = './dataset/train/' + cl + "/" + file
                shutil.move(src_path, dst_path)

    epc = int(input("Enter the number of epochs: "))
    train(epc)
