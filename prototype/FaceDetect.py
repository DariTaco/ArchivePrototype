"""
Face Detection
Sources: https://github.com/ageitgey/face_recognition

haar model: https://github.com/opencv/opencv/tree/master/data/haarcascades

https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector:
caffe model: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
             https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
tf model: https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180220_uint8/opencv_face_detector_uint8.pb
          https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt
"""

import PIL.Image
import PIL.ImageDraw
import face_recognition as fr
import time
import cv2 as cv


# define functions
def dlib_hog(path, upsample):
    global face_locations, start, end, draw_image, image
    image = fr.load_image_file(path)
    draw_image = PIL.Image.fromarray(image)

    start = time.time()
    face_locations = fr.face_locations(image, number_of_times_to_upsample=upsample)
    end = time.time()


def dlib_cnn(path, upsample):
    global face_locations, start, end, draw_image, image
    image = fr.load_image_file(path)
    draw_image = PIL.Image.fromarray(image)

    start = time.time()
    face_locations = fr.face_locations(image, number_of_times_to_upsample=upsample, model="cnn")
    end = time.time()


def opencv_haar(path):
    global face_locations, start, end, draw_image, image
    face_cascade = cv.CascadeClassifier('face detection models/haarcascade_frontalface_default.xml')

    image = cv.imread(path)
    draw_image = PIL.Image.fromarray(image)

    start = time.time()
    face_locations = face_cascade.detectMultiScale(image, minNeighbors=5)
    end = time.time()


def opencv_tensorflow(path):
    global face_locations, start, end, draw_image, image
    model_file = "face detection models/TFmodel.pb"
    config_file = "face detection models/TFconfig.pbtxt"
    dnn = cv.dnn.readNetFromTensorflow(model_file, config_file)

    image = cv.imread(path)
    draw_image = PIL.Image.fromarray(image)
    blob = cv.dnn.blobFromImage(image)
    dnn.setInput(blob)

    start = time.time()
    face_locations = dnn.forward()
    end = time.time()


def opencv_caffe(path):
    global face_locations, start, end, draw_image, image
    model_file = "face detection models/res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "face detection models/caffeDeploy.prototxt"
    dnn = cv.dnn.readNetFromCaffe(config_file, model_file)

    image = cv.imread(path)
    blob = cv.dnn.blobFromImage(image)
    dnn.setInput(blob)
    draw_image = PIL.Image.fromarray(image)

    start = time.time()
    face_locations = dnn.forward()
    end = time.time()


''' 
    every detection method returns different parameters or data structures. 
    Hence 3 methods (drawBoxes 1,2 and 3)
'''


# for dlib CNN and HOG
def draw_boxes1():
    global face_locations, face_count, draw_boxes, new_locations
    draw_boxes = "draw_boxes1"
    face_count = 0
    new_locations = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        new_locations.append([top, right, bottom, left])
        face_count = face_count + 1
        # draw on image
        draw = PIL.ImageDraw.Draw(draw_image)
        draw.rectangle([left, top, right, bottom], outline="red")
    return draw_image


# for OpenCV Haar
def draw_boxes2():
    global face_locations, face_count, new_locations
    face_count = 0
    new_locations = []
    for face_location in face_locations:
        x, y, w, h = face_location
        left, top, right, bottom = x, y, x + w, y + h
        new_locations.append([top, right, bottom, left])
        face_count = face_count + 1
        # draw on image
        draw = PIL.ImageDraw.Draw(draw_image)
        draw.rectangle([left, top, right, bottom], outline="red")
    return draw_image


# for OpenCV CNN Tensorflow and CNN caffe
def draw_boxes3():
    global face_locations, face_count, image, new_locations
    face_count = 0
    tresh = 0.3
    (height, width) = image.shape[:2]
    new_locations = []
    for i in range(face_locations.shape[2]):
        confidence = face_locations[0, 0, i, 2]
        if confidence > tresh:
            face_count = face_count + 1
            left = int(face_locations[0, 0, i, 3] * width)
            top = int(face_locations[0, 0, i, 4] * height)
            right = int(face_locations[0, 0, i, 5] * width)
            bottom = int(face_locations[0, 0, i, 6] * height)
            new_locations.append([top, right, bottom, left])
            # draw on image
            draw = PIL.ImageDraw.Draw(draw_image)
            draw.rectangle([left, top, right, bottom], outline="red")
    return draw_image


# get the time it took to detect the faces
def get_detection_time():
    return end - start


# get the number of faces detected
def get_number_of_faces_detected():
    return face_count


# get locations of detected faces
def get_new_locations():
    return new_locations
