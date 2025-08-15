import sys
import importlib

try:
    import tensorflow.keras.preprocessing.image
except ModuleNotFoundError:
    keras_preprocessing_image = importlib.import_module("keras.preprocessing.image")
    sys.modules["tensorflow.keras.preprocessing.image"] = keras_preprocessing_image
