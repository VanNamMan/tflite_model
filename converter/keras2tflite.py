import tensorflow as tf
from keras.models import load_model
import sys

# print(sys.path)

print(tf.__version__)
file_name = "D:/Python/Pretrained/face_net/facenet_keras.h5"
# model = load_model(file_name)

# converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(file_name)
# converter.optimizations = [tf.contrib.lite.optimizations.]
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)