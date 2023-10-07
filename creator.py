import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
