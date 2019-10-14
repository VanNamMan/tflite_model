import tensorflow as tf
import numpy as np
import pandas as pd
import os

print(tf.__version__)

input_height = 300
input_width = 300

def load_saved_model(path):
    the_graph = tf.Graph()
    with tf.Session(graph=the_graph) as sess:
        tf.saved_model.loader.load(sess, 
                [tf.saved_model.tag_constants.SERVING], path)
    return the_graph

def get_anchors(graph, tensor_name):
    image_tensor = graph.get_tensor_by_name("image_tensor:0")
    box_corners_tensor = graph.get_tensor_by_name(tensor_name)
    box_corners = sess.run(box_corners_tensor, feed_dict={
        image_tensor: np.zeros((1, input_height, input_width, 3))})

    return np.stack(box_corners)

##    ymin, xmin, ymax, xmax = np.transpose(box_corners)    
##    width = xmax - xmin
##    height = ymax - ymin
##    ycenter = ymin + height / 2.
##    xcenter = xmin + width / 2.
##    return np.stack([ycenter, xcenter, height, width])

saved_model_path = "ssdlite_mobilenet_v2_coco_2018_05_09/saved_model"
print(os.listdir(saved_model_path))
the_graph = load_saved_model(saved_model_path)

anchors_tensor = "Concatenate/concat:0"
with the_graph.as_default():
    with tf.Session(graph=the_graph) as sess:
        anchors = get_anchors(the_graph, anchors_tensor)

print(type(anchors))
print(anchors.shape)

dict_anchor = {}
dict_anchor["ymin"] = anchors[:,0]
dict_anchor["xmin"] = anchors[:,1]
dict_anchor["ymax"] = anchors[:,2]
dict_anchor["xmax"] = anchors[:,3]

df = pd.DataFrame(dict_anchor,columns=["ymin","xmin","ymax","xmax"])
df.to_csv("mobile_ssd_v2_anchor.csv")


