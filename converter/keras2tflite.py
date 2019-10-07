import tensorflow as tf
print(tf.__version__)
file_name = "D:/Python/Pretrained/face_net/facenet_keras.h5"
converter = tf.lite.TFLiteConverter.from_keras_model_file(file_name)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)