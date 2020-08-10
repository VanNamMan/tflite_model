from libs.utils import *

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tflite/coco_ssd_mobilenet/detect.tflite")
interpreter.allocate_tensors()

# 
labels = load_labels("tflite/coco_ssd_mobilenet/labelmap.txt")
colors = np.random.uniform(0,255,(len(labels),3))
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

#
cap = cv2.VideoCapture("data/dog_cat.mp4")
t0 = time.time()
n_frame = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    ret,x = cap.read()
    if not ret:
        break
    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
    x = cv2.resize(x,(input_width,input_height))

    # t0 = time.time()
    results = detect_objects(interpreter, x, 0.3)
    n_frame += 1
    dt = time.time() - t0
    fps = n_frame/dt
    # print("find : %d object, time : %.2f"%(len(results),dt))

    image = draw_object_detection_results(x,results,labels,colors)
    image = cv2.putText(image,'%.2f'%fps,(20, 20),font,1,(0,255,0),2)
    cv2.imshow("",image)
    cv2.waitKey(1)

cv2.destroyAllWindows()










