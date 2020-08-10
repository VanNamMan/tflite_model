import numpy as np
import tensorflow as tf
import time
import cv2

# load MobileNetV2 labels
def load_label(label_path):
    labels = [l.strip() for l in open(label_path,"r").readlines()]
    return labels

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tflite/mobilenet_v2/detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = load_label("tflite/mobilenet_v2/labelmap.txt")
indexs = np.arange(1000)

# 
while True:
    filename = input("image path : ")
    if not filename:
        break
    x = cv2.imread(filename)
    if x is None:
        continue
    print("Load image : ",x.shape)
    x = cv2.resize(x,(224,224))
    x = np.array([x]).astype(np.float32)/255

    # Test the TensorFlow Lite model on random input data.
    t0 = time.time()
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    # input_data = x
    print("Input data : ",x.shape)
    interpreter.set_tensor(input_details[0]['index'], x)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    tflite_results = interpreter.get_tensor(output_details[0]['index'])[0]
    res, ids = zip(*sorted(zip(tflite_results, indexs),reverse=True))
    print("====> tflite: ")
    for i in range(3):
        print(" %d,%s,%.2f"%(ids[i],labels[ids[i]],res[i]))

    print("time (tflite) : ",round(time.time() - t0,2))









