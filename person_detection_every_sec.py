# ----------------------------------------------
# --- Author         : Pushpraj Katiyar
# --- Mail           : pushprajkatiyar@gmail.com
# --- Date           : 26th April 2019
# ----------------------------------------------
# --- Author         : Manfred Weber
# --- Mail           : manfred.weber@gmail.com
# --- Date           : 1th August 2021
# ----------------------------------------------

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
import zipfile
import json
import argparse
import cv2

if tf.__version__ < '1.13.0':
    raise ImportError(
        'Please upgrade your tensorflow installation to v1.13.* or later!')

tf.disable_v2_behavior()

sys.path.insert(0, 'utils')
import label_map_util
import people_class_util as class_utils
import visualization_utils as vis_util

# What model to download. KEEP IN MIND ssd_mobilenet_v1_coco_2018_01_28 is fastest model but low accuracy
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
MODEL_NAME = 'faster_rcnn_nas_lowproposals_coco_2017_11_08'
# MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
# MODEL_NAME = 'faster_rcnn_resnet50_lowproposals_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'utils/person_label_map.pbtxt'

#cap = cv2.VideoCapture('http://devimages.apple.com/iphone/samples/bipbop/bipbopall.m3u8')
# checking if file exist or not and if yes then downloading specified model
if not os.path.exists(MODEL_FILE):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

NUM_CLASSES = 50

# loading specified class/category description
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# some helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# input is mp4
# output is avi
# avi then a layer on top of the mp4 (in your video editor) with composition = screen
filename = "DEMO-WALK-FINAL"

# start providing video
# you can use live CCTV video URL in place of "test_video.mp4"
cap = cv2.VideoCapture(filename+'.mp4')

# variables
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
out = cv2.VideoWriter(filename+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), frame_rate, size)
img_height, img_width = frame_height, frame_width
frame_count = 0
second_completed = 0
person_count = 0
alpha = 1.0
n_channels = 3
transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
info_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)
black_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

# video detection
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
        while(True):

            #read every frame
            success, image_np = cap.read()
            if not success:
                print ('>>>>>  End of the video file...')
                break
            frame_count += 1

            #check if this frame is closed to time interval provided
            if frame_count == (1 * frame_rate):
                second_completed += 1
                print('>>>>>' + str(second_completed))
                frame_count = 0

                alpha = 1.0
                n_channels = 3
                transparent_img = np.zeros((img_height, img_width, n_channels), dtype=np.uint8)

                #flaten image using numpy
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.

                vis_util.visualize_boxes_and_labels_on_image_array(
                    transparent_img,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=10)
                #creating annotation array which can be sent to API for further operations
                annotations = {}
                annotations['class_annotations'], count = (
                    class_utils.get_class(np.squeeze(classes).astype(np.int32),
                                          category_index, np.squeeze(boxes),
                                          np.squeeze(scores)))
                annotations['person_count'] = count
                person_count = person_count + count

            # fade out visualization
            if alpha > 0:
                alpha = alpha - 0.2
                alpha_b = 1.0 - alpha


            result = cv2.addWeighted(transparent_img, alpha, black_img, alpha_b, 0.0)
            cv2.putText( result, str(person_count), (50,150), cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0, 255), 6)
            cv2.imshow('Preview', result)
            out.write(result)

            # allow break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                break

# Release the capture

cap.release()
out.release()
cv2.destroyAllWindows()
