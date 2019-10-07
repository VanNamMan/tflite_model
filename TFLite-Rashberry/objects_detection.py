from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
import numpy as np
import cv2
import os,re
print(os.getcwd())

import pickle

from tflite_runtime.interpreter import Interpreter

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

def convert_boxs(box,size):
    w,h = size
    top,left,bot,right = box
    top = int(h*top)
    left = int(w*left)
    bot = int(h*bot)
    right = int(w*right)
    tl = (left,top)
    br = (right,bot)
    return tl,br

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': int(classes[i]),
          'score': scores[i]
      }
      results.append(result)
  return results


def main():
#    parser = ArgumentParser(description='Demo Camera')
#    parser.add_argument('-t','--threshold', type=float,help='threshold for classification')
#    parser.add_argument('-m','--model_path', type=str,help='threshold for classification')
#    args = parser.parse_args()
#    
#    threshold = args.threshold
#    model_path = args.model_path

    interpreter = Interpreter(model_path="./detect/detect.tflite")
    interpreter.allocate_tensors()
    
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

    labels = load_labels("./detect/labelmap.txt")
    print("n-class : ",len(labels))
    print(labels)
    offset_label = 1
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():

        ret,image = cap.read()
        img = cv2.resize(image,(input_width, input_height))
        
        results = detect_objects(interpreter,img,0.4)
        
        for res in results:
            box = res["bounding_box"]
            class_id = res["class_id"]
            tl,br = convert_boxs(box,size=(CAMERA_WIDTH,CAMERA_HEIGHT))
            cv2.rectangle(image,tl,br,(0,255,0),2)
            cv2.putText(image,labels[class_id+offset_label],tl,0,1,(0,255,255),2)
    
        cv2.imshow("",image)
        if cv2.waitKey(22) == ord('q'):
            break
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()
