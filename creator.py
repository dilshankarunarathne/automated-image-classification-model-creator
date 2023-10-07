import os
import tensorflow as tf        # tensorflow library
from tensorflow import keras 
from tensorflow.keras import layers  # type of layers

path_to_plants = "dataset"
plant_classes = os.listdir(path_to_plants)

