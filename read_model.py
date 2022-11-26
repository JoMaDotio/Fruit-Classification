import numpy as np
import tensorflow as tf
from tensorflow import keras


saved_model = keras.models.load_model('#theFileName')


saved_model.summary()
