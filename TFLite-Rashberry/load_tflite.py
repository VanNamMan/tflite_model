from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser
import numpy as np
import cv2
import os
print(os.getcwd())

import pickle

from tflite_runtime.interpreter import Interpreter

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def load_svcModel(folder):
    encoder = pickle.load(open(folder+"/encoder.pickle","rb"))
    clf = pickle.load(open(folder+"/svc_model.pickle","rb"))
    return clf,encoder

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

def main():
    parser = ArgumentParser(description='Demo Camera')
    parser.add_argument('-t','--threshold', type=float,help='threshold for classification')
    parser.add_argument('-m','--model_path', type=str,help='threshold for classification')
    args = parser.parse_args()
    
    threshold = args.threshold
    model_path = args.model_path

    interpreter = Interpreter(model_path="%s/face_net.tflite"%model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ##
    print("===> input : ",type(input_details),input_details)
    print("===> output : ",type(output_details),output_details)
    ##
    ### check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    print("floating_model : ",floating_model)
    ##
    ### NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    print("input : ",input_details[0]['shape'])
    ##
    ##labels = load_labels("/home/pi/Downloads/detect/labelmap.txt")
    ##print("n-class : ",len(labels))
    ### print(labels)
    ##
    clf,encoder = load_svcModel("%s/model"%model_path)
    print(clf)
    print(encoder)
    face_cascade = cv2.CascadeClassifier('%s/model/haarcascade_frontalface_default.xml'%model_path)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # image = cv2.imread("/home/pi/Downloads/dog.jpg")
        ret,image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.3,minNeighbors = 5
                                    , minSize = (70,70))
        
        for (x,y,w,h) in faces:
            
            roi = crop(image,[x,y,w,h],image_size=160,margin=10)
    ##        roi = image[y-10:y+h+10,x-10:x+w+10]
    ##        roi = cv2.resize(roi,(width, height))
            roi = prewhiten(roi)
            
            input_data = np.expand_dims(roi, axis=0).astype(np.float32)
    ##        print(input_data.dtype)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred,proba = infer(encoder,clf,l2_normalize(output_data))
            
            pred,score = pred[0],proba.max()
            if score > threshold:
                print(pred,score)
                cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(image,pred + ",%.2f"%score,(x,y),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
            
                    
    ##    img = cv2.resize(image,(width, height))
    ##    # if not ret :
    ##    #     break
    ##    # add N dim
    ##    input_data = np.expand_dims(img, axis=0)
    ##    # print(type(input_data))
    ##
    ##    if floating_model:
    ##        input_data = (np.float32(input_data) - 128) / 128
    ##
    ##    interpreter.set_tensor(input_details[0]['index'], input_data)
    ##
    ##    interpreter.invoke()
    ##
    ##    output_data = interpreter.get_tensor(output_details[0]['index'])
    ##    
    ##    print(output_data)
    ##    boxs = np.squeeze(output_data)
    ##    # print("boxs : ",boxs.shape)
    ##
    ##    class_id = interpreter.get_tensor(output_details[1]['index'])
    ##    class_id = np.squeeze(class_id)
    ##    # print("class_id : ",class_id)
    ##
    ##    scores = interpreter.get_tensor(output_details[2]['index'])
    ##    scores = np.squeeze(scores)
    ##    # print("scores : ",scores.shape,scores)
    ##
    ##    numbers = interpreter.get_tensor(output_details[3]['index'])
    ##    numbers = np.squeeze(numbers)
    ##    # print("numbers : ",numbers)
    ##
    ##    idx = np.where(scores>0.4)[0]
    ##    # print(idx)
    ##    size = image.shape[:2][::-1]
    ##    for i in idx:
    ##        box = boxs[i]
    ##        tl,br = convert_boxs(box,size=size)
    ##        cv2.rectangle(image,tl,br,(0,255,0),2)
    ##        cv2.putText(image,labels[int(class_id[i])],tl,0,1,(0,255,255),2)
    ##
        cv2.imshow("",image)
        if cv2.waitKey(22) == ord('q'):
            break
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    main()
