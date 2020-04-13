"""
Object Detection
Sources: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
yolo model: https://pjreddie.com/darknet/yolo/
tf models: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

"""

import numpy as np
import tensorflow as tf
import cv2 as cv
import PIL.Image
import PIL.ImageDraw
import time
import ObjectLists

CONFIDENCE_THRESHOLD = 0.5


def faster_rcnn_oi(path):
    tf.reset_default_graph()

    # read the cnn
    with tf.io.gfile.GFile('object detection models/faster_rcnn_oi.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess1:
        # Restore session
        sess1.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # read the image, run the model and draw the bounding boxes
        read_image(path)
        run_model(sess1)
        draw_boxes("open images")

    return draw_image


def faster_rcnn_coco(path):
    tf.reset_default_graph()

    # read the cnn
    with tf.io.gfile.GFile('object detection models/faster_rcnn_coco.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess2:
        # Restore session
        sess2.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # read the image, run the model and draw the bounding boxes
        read_image(path)
        run_model(sess2)
        draw_boxes("coco")

    return draw_image


def mask_rcnn_coco(path):
    tf.reset_default_graph()

    # read the cnn
    with tf.io.gfile.GFile('object detection models/mask_rcnn_coco.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess3:
        # Restore session
        sess3.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # read the image, run the model and draw the bounding boxes
        read_image(path)
        run_model(sess3)
        draw_boxes("coco")

    return draw_image


def ssd_inception_coco(path):
    tf.reset_default_graph()

    # read the cnn
    with tf.io.gfile.GFile('object detection models/ssd_inception_coco.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess4:
        # Restore session
        sess4.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # read the image, run the model and draw the bounding boxes
        read_image(path)
        run_model(sess4)
        draw_boxes("coco")

    return draw_image


def ssd_resnet50_coco(path):
    tf.reset_default_graph()

    # read the cnn
    with tf.io.gfile.GFile('object detection models/ssd_resnet50_coco.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess5:
        # Restore session
        sess5.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # read the image, run the model and draw the bounding boxes
        read_image(path)
        run_model(sess5)
        draw_boxes("coco")

    return draw_image


# yolo has to be run a bit differently
def yolo_v3_coco(path):
    tf.reset_default_graph()

    # load model
    model_file = "object detection models/yolov3.weights"
    config_file = "object detection models/yolov3.cfg"
    dnn = cv.dnn.readNetFromDarknet(config_file, model_file)

    # read the image, run the model and draw the bounding boxes
    read_image(path)
    run_model_yolo_v3(dnn)
    draw_boxes_yolo_v3("coco")

    return draw_image


def yolo_v3_oi(path):
    tf.reset_default_graph()

    # load model
    model_file = "object detection models/yolov3oi.weights"
    config_file = "object detection models/yolov3oi.cfg"
    dnn = cv.dnn.readNetFromDarknet(config_file, model_file)

    # read the image, run the model and draw the bounding boxes
    read_image(path)
    run_model_yolo_v3(dnn)
    draw_boxes_yolo_v3("open images")

    return draw_image


# helping methods
def read_image(path):
    global image, draw_image, height, width, inp
    image = cv.imread(path)
    draw_image = PIL.Image.fromarray(image)
    (height, width) = image.shape[:2]
    inp = image
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB


# runs the yolo model
def run_model_yolo_v3(dnn):
    global out, start, end
    layer_names = dnn.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in dnn.getUnconnectedOutLayers()]
    # yolo only takes 320x320, 416x416 or 608x608
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, size=(608, 608), swapRB=True, crop=False)
    dnn.setInput(blob)
    start = time.time()
    out = dnn.forward(layer_names)
    end = time.time()


# runs the tensorflow model
def run_model(sess):
    global out, start, end
    start = time.time()
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    end = time.time()
    sess.close()


def draw_boxes_yolo_v3(trainingsdataset):
    global number_of_objects_detected, detected_objects
    number_of_objects_detected = 0
    detected_objects = []
    color = "red"

    for o in out:
        for detection in o:

            scores = detection[5:]
            class_id = np.argmax(scores)
            score = scores[class_id]

            if score > CONFIDENCE_THRESHOLD:
                if trainingsdataset == "coco":
                    obj = ObjectLists.get_yolo_v3_coco_object_by_classID(class_id)
                elif trainingsdataset == "open images":
                    obj = ObjectLists.get_oi_object_by_classID(class_id + 1)
                rounded_score_in_percent = 100 * (round(score, 4))
                detected_objects.append("{} {:.2f}%".format(obj, rounded_score_in_percent))
                number_of_objects_detected += 1

                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(center_x - (w / 2))
                y = int(center_y - (h / 2))

                left, top, right, bottom = x, y, x + w, y + h
                draw = PIL.ImageDraw.Draw(draw_image)
                draw.rectangle([left, top, right, bottom], outline=color)
                draw.text([left, bottom], obj, fill=color)


# draws the bounding boxes for the tensorflow model
def draw_boxes(trainingsdataset):
    global number_of_objects_detected, detected_objects

    detected_objects = []
    number_of_objects_detected = 0
    number_of_total_detections = int(out[0][0])
    color = "red"

    for i in range(number_of_total_detections):
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > CONFIDENCE_THRESHOLD:

            # append to list of detected objects
            class_id = int(out[3][0][i])
            if trainingsdataset == "coco":
                obj = ObjectLists.get_coco_object_by_classID(class_id)
            elif trainingsdataset == "open images":
                obj = ObjectLists.get_oi_object_by_classID(class_id)
            rounded_score_in_percent = 100 * (round(score, 4))
            detected_objects.append("{} {:.2f}%".format(obj, rounded_score_in_percent))
            number_of_objects_detected += 1

            # draw bounding box and object name
            left = bbox[1] * width
            top = bbox[0] * height
            right = bbox[3] * width
            bottom = bbox[2] * height
            draw = PIL.ImageDraw.Draw(draw_image)
            draw.rectangle([left, top, right, bottom], outline=color)
            draw.text([left, bottom], obj, fill=color)


# get the time it took to detect the objects
def get_detection_time():
    global end, start
    return end - start


# get the number of objects detected
def get_number_of_objects_detected():
    return number_of_objects_detected


# returns a list of all the objects that have been detected
def get_detected_objects():
    return detected_objects
