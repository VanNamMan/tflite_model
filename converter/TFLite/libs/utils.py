import numpy as np
import tensorflow as tf
import time,re
import cv2

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

# def load_labels(label_path):
#     labels = [l.strip() for l in open(label_path,"r").readlines() if "???" not in l]
#     return labels

def set_input_tensor(interpreter,image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter,index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter,image, threshold):
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
          'class_id': int(classes[i]) + 1,
          'score': scores[i]
      }
      results.append(result)
  return results

def draw_object_detection_results(image, results, labels,colors):
  """Draws the bounding box and label for each object in the results."""
  font = cv2.FONT_HERSHEY_COMPLEX_SMALL
  height,width = image.shape[:2]
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)

    # Overlay the box, label, and score on the camera preview
    cv2.rectangle(image,(xmin, ymin),(xmax, ymax),colors[obj['class_id']],2)
    cv2.putText(image,'%s,%.2f' % (labels[obj['class_id']], obj['score']),(xmin, ymin),font,1,colors[obj['class_id']],2)
  return image