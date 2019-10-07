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


def crop(img,box,image_size=160,margin=10):

    h0,w0 = img.shape[:2]
    x,y,w,h = box

    x1 = max(0,x-margin//2)
    y1 = max(0,y-margin//2)

    x2 = min(x+w+margin//2,w0)
    y2 = min(y+h+margin//2,h0)

    cropped = img[y1:y2,x1:x2,:]
    aligned = cv2.resize(cropped, (image_size, image_size),interpolation=cv2.INTER_CUBIC)
    return aligned

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def infer(le=None, clf=None,embs=None):
    """
    le : labels Encoder
    clf : svc model
    """
    pred = le.inverse_transform(clf.predict(embs))
    proba = clf.predict_proba(embs)
    return pred,proba

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

def detect_face(interpreter,face):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, face)
  interpreter.invoke()

  # Get all output details
  embedding = get_output_tensor(interpreter, 0)
  print(embedding.shape)
  pred,proba = infer(encoder,clf,[l2_normalize(embedding)])
  
  result = {
          'name': pred[0],
          'score': proba.max()
      }
  return result

if __name__ == "__main__":
    
    threshold = 0.5
    # svc classifier
    encoder = pickle.load(open("./svc/encoder.pickle","rb"))
    clf = pickle.load(open("./svc/svc_model.pickle","rb"))
    print(clf)
    print(encoder)
    # face detection
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # load interper face_net model
    interpreter = Interpreter(model_path="./face_net/face_net.tflite")
    interpreter.allocate_tensors()
    
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    
    # real time 
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret,image = cap.read()
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.3,minNeighbors = 5
                                    , minSize = (70,70))
        
        for (x,y,w,h) in faces:
            roi = crop(image,[x,y,w,h],image_size=160,margin=10)
            roi = cv2.resize(roi,(input_width, input_height))
            roi = prewhiten(roi)
        
            result = detect_face(interpreter,roi)
            score = result["score"]
            pred = result["name"]
            
            if score > threshold:
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(image,pred + ",%.2f"%score,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                
        cv2.imshow("",image)
        if cv2.waitKey(22) == ord('q'):
            break
    cv2.destroyAllWindows()
            
            
    
    
    
    
    
    
    
    