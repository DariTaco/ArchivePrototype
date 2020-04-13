"""
UI and UI interaction methods
"""

import sys
from PySide2 import QtCore, QtGui, QtWidgets
import FaceDetect
import FaceRecog
import ObjectDetect
import cv2

app = QtWidgets.QApplication(sys.argv)


# define helping methods
def reset_results():
    list_recognized_faces.clear()
    repaint_list(list_recognized_faces)
    list_detected_objects.clear()
    repaint_list(list_recognized_faces)
    l_objects_detected.setText("0")
    l_faces_detected.setText("0")
    l_faces_recognized.setText("0")
    l_object_detection_time.setText("0.00000")
    l_face_detection_time.setText("0.00000")
    l_face_recognition_time.setText("0.00000")


def repaint_list(l):
    # bugfix: repaint doesn't work if only the text of the items changes
    l.repaint()
    l.hide()
    l.show()


def display_list_detected_objects():
    list_detected_objects.clear()
    detected_objects = ObjectDetect.get_detected_objects()
    for item in detected_objects:
        list_detected_objects.addItem(str(item))

    repaint_list(list_detected_objects)


def display_list_recognized_faces():
    list_recognized_faces.clear()
    recognized_faces = FaceRecog.get_recognized_faces()
    for item in recognized_faces:
        list_recognized_faces.addItem(item)

    repaint_list(list_recognized_faces)


def display_object_detection_time():
    object_detection_time = str(round(ObjectDetect.get_detection_time(), 5))
    l_object_detection_time.setText(object_detection_time)
    l_object_detection_time.repaint()


def display_number_of_objects_detected():
    number_of_objects_detected = str(ObjectDetect.get_number_of_objects_detected())
    l_objects_detected.setText(number_of_objects_detected)
    l_objects_detected.repaint()


def display_detection_time():
    detection_time = str(round(FaceDetect.get_detection_time(), 5))
    l_face_detection_time.setText(detection_time)
    l_face_detection_time.repaint()


def display_number_of_faces_detected():
    number_of_faces_detected = str(FaceDetect.get_number_of_faces_detected())
    l_faces_detected.setText(number_of_faces_detected)
    l_faces_detected.repaint()


def display_image_faces(image_path):
    qimage = QtGui.QImage(image_path)
    qimage = qimage.scaled(600, 600)
    pixmap = QtGui.QPixmap.fromImage(qimage)
    l_uploaded_image_faces.setPixmap(pixmap)
    l_uploaded_image_faces.repaint()


def display_image_objects(image_path):
    qimage = QtGui.QImage(image_path)
    qimage = qimage.scaled(600, 600)
    pixmap = QtGui.QPixmap.fromImage(qimage)
    l_uploaded_image_objects.setPixmap(pixmap)
    l_uploaded_image_objects.repaint()


def display_recognition_time():
    recognition_time = str(round(FaceRecog.get_recognition_time(), 5))
    l_face_recognition_time.setText(recognition_time)
    l_face_recognition_time.repaint()


def display_number_of_faces_recognized():
    number_of_faces_recognized = str(FaceRecog.get_number_of_faces_recognized())
    l_faces_recognized.setText(number_of_faces_recognized)
    l_faces_recognized.repaint()


def normalize_image(image_path):
    image = cv2.imread(image_path)
    # convert to gray, add padding if necessary and scale to 600x600
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape[:2]
    if w > h:
        # add padding bottom and top
        pad = int((w - h) / 2)
        image = cv2.copyMakeBorder(image, pad, pad,
                                   0, 0, cv2.BORDER_CONSTANT,
                                   None, (255, 255, 255))
    elif h > w:
        # add padding left and right
        pad = int((h - w) / 2)
        image = cv2.copyMakeBorder(image, 0, 0,
                                   pad, pad, cv2.BORDER_CONSTANT,
                                   None, (255, 255, 255))
    image = cv2.resize(image, (600, 600))
    return image


# define methods
def pb_recognize_faces_clicked():
    # run selected recognition method
    recognition_method = cb_recognition_method.currentText()
    number_of_neighbors = int(recognition_method[:1])
    recognize_image = FaceRecog.run_knn(upload_image_path, number_of_neighbors)

    # display image with detected objects
    recognize_image.save("temp files/tempRecog.jpg")
    display_image_faces("temp files/tempRecog.jpg")

    # display results
    display_recognition_time()
    display_number_of_faces_recognized()
    display_list_recognized_faces()


def pb_detect_objects_clicked():
    # run selected detection method
    detection_method = cb_object_detection_method.currentText()
    if detection_method == "Faster-RCNN (OpenImages)":
        detect_image_objects = ObjectDetect.faster_rcnn_oi(upload_image_path)
    elif detection_method == "Faster-RCNN (Coco)":
        detect_image_objects = ObjectDetect.faster_rcnn_coco(upload_image_path)
    elif detection_method == "Mask-RCNN (Coco)":
        detect_image_objects = ObjectDetect.mask_rcnn_coco(upload_image_path)
    elif detection_method == "SSD Inception (Coco)":
        detect_image_objects = ObjectDetect.ssd_inception_coco(upload_image_path)
    elif detection_method == "SSD Resnet50 (Coco)":
        detect_image_objects = ObjectDetect.ssd_resnet50_coco(upload_image_path)
    elif detection_method == "YOLOv3 (OpenImages)":
        detect_image_objects = ObjectDetect.yolo_v3_oi(upload_image_path)
    elif detection_method == "YOLOv3 (Coco)":
        detect_image_objects = ObjectDetect.yolo_v3_coco(upload_image_path)

    # display image with detected objects
    detect_image_objects.save("temp files/tempObjects.jpg")
    display_image_objects("temp files/tempObjects.jpg")

    # display results
    display_object_detection_time()
    display_number_of_objects_detected()
    display_list_detected_objects()


def pb_detect_faces_clicked():
    # run selected detection method
    detection_method = cb_face_detection_method.currentText()
    if detection_method == "Dlib Hog":
        FaceDetect.dlib_hog(upload_image_path, 1)
        detect_image = FaceDetect.draw_boxes1()
    elif detection_method == "Dlib CNN":
        FaceDetect.dlib_cnn(upload_image_path, 1)
        detect_image = FaceDetect.draw_boxes1()
    elif detection_method == "OpenCV Haar default.xml":
        FaceDetect.opencv_haar(upload_image_path)
        detect_image = FaceDetect.draw_boxes2()
    elif detection_method == "OpenCV CNN tensorflow":
        FaceDetect.opencv_tensorflow(upload_image_path)
        detect_image = FaceDetect.draw_boxes3()
    elif detection_method == "OpenCV CNN caffe":
        FaceDetect.opencv_caffe(upload_image_path)
        detect_image = FaceDetect.draw_boxes3()
    elif detection_method == "Dlib Hog2":
        FaceDetect.dlib_hog(upload_image_path, 2)
        detect_image = FaceDetect.draw_boxes1()
    elif detection_method == "Dlib CNN2":
        FaceDetect.dlib_cnn(upload_image_path, 2)
        detect_image = FaceDetect.draw_boxes1()

    # display image with detected faces
    detect_image.save("temp files/temp.jpg")
    display_image_faces("temp files/temp.jpg")

    # display results
    reset_results()
    display_detection_time()
    display_number_of_faces_detected()

    # enable recognizing faces
    pb_recognize_faces.setDisabled(False)
    pb_recognize_faces.repaint()


def pb_upload_image_clicked():
    global upload_image_path

    # file dialog
    fd_upload_image = QtWidgets.QFileDialog()
    fd_upload_image.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    fd_upload_image.setNameFilters(["Images (*.jpg)"])
    fd_upload_image.selectNameFilter("Images (*.jpg)")
    fd_upload_image.exec_()
    files = fd_upload_image.selectedFiles()

    # if image was uploaded
    if files:
        upload_image_path = files[0]

        # normalize image
        image = normalize_image(upload_image_path)
        cv2.imwrite("temp files/normalizeTemp.jpg", image)
        upload_image_path = "temp files/normalizeTemp.jpg"

        # display image
        display_image_faces(upload_image_path)
        display_image_objects(upload_image_path)

        # enable detecting faces and detecting objects
        pb_detect_faces.setDisabled(False)
        pb_detect_faces.repaint()
        pb_detect_objects.setDisabled(False)
        pb_detect_objects.repaint()

        # disable recognizing faces
        pb_recognize_faces.setDisabled(True)

        # clear results from previous detections
        reset_results()


###########FACES TAB UI start############
# define widgets
l_title_faces = QtWidgets.QLabel("FACES")
l_title_faces.setFont(QtGui.QFont("Courier", 38, QtGui.QFont.Bold))

l_label1_faces = QtWidgets.QLabel("detection method:")
l_label2_faces = QtWidgets.QLabel("recognition method:")
l_label3_faces = QtWidgets.QLabel("detection time:")
l_label4_faces = QtWidgets.QLabel("faces detected:")
l_label5_faces = QtWidgets.QLabel("recognition time:")
l_label6_faces = QtWidgets.QLabel("faces recognized:")
l_face_detection_time = QtWidgets.QLabel("0.00000")
l_faces_detected = QtWidgets.QLabel("0")
l_face_recognition_time = QtWidgets.QLabel("0.00000")
l_faces_recognized = QtWidgets.QLabel("0")
l_uploaded_image_faces = QtWidgets.QLabel("")
display_image_faces("white600x600.jpg")


pb_detect_faces = QtWidgets.QPushButton("Detect Faces")
pb_detect_faces.clicked.connect(pb_detect_faces_clicked)
pb_detect_faces.setDisabled(True)
pb_recognize_faces = QtWidgets.QPushButton("Recognize Faces")
pb_recognize_faces.clicked.connect(pb_recognize_faces_clicked)
pb_recognize_faces.setDisabled(True)
pb_upload_image = QtWidgets.QPushButton("Upload Image ...")
pb_upload_image.clicked.connect(pb_upload_image_clicked)
pb_quit = QtWidgets.QPushButton("Quit Application")
QtCore.QObject.connect(pb_quit, QtCore.SIGNAL("clicked()"), app, QtCore.SLOT("quit()"))  # for quitting the application

cb_face_detection_method = QtWidgets.QComboBox()
cb_face_detection_method.addItem("Dlib Hog")
cb_face_detection_method.addItem("Dlib Hog2")
cb_face_detection_method.addItem("Dlib CNN")
cb_face_detection_method.addItem("Dlib CNN2")
cb_face_detection_method.addItem("OpenCV Haar default.xml")
cb_face_detection_method.addItem("OpenCV CNN tensorflow")
cb_face_detection_method.addItem("OpenCV CNN caffe")
cb_recognition_method = QtWidgets.QComboBox()
cb_recognition_method.addItem("1 KNN")
cb_recognition_method.addItem("2 KNN")
cb_recognition_method.addItem("3 KNN")
cb_recognition_method.addItem("4 KNN")
cb_recognition_method.addItem("5 KNN")

list_recognized_faces = QtWidgets.QListWidget()

# define windows
window_faces = QtWidgets.QWidget()
window1_faces = QtWidgets.QWidget()
window2_faces = QtWidgets.QWidget()
window3_faces = QtWidgets.QWidget()
window4_faces = QtWidgets.QWidget()
window5_faces = QtWidgets.QWidget()
window6_faces = QtWidgets.QWidget()
window7_faces = QtWidgets.QWidget()
window8_faces = QtWidgets.QWidget()
window9_faces = QtWidgets.QWidget()
window10_faces = QtWidgets.QWidget()

# define layouts
layout_faces = QtWidgets.QVBoxLayout()
layout1_faces = QtWidgets.QHBoxLayout()
layout2_faces = QtWidgets.QVBoxLayout()
layout3_faces = QtWidgets.QVBoxLayout()
layout4_faces = QtWidgets.QVBoxLayout()
layout5_faces = QtWidgets.QHBoxLayout()
layout6_faces = QtWidgets.QHBoxLayout()
layout7_faces = QtWidgets.QHBoxLayout()
layout8_faces = QtWidgets.QHBoxLayout()
layout9_faces = QtWidgets.QHBoxLayout()
layout10_faces = QtWidgets.QHBoxLayout()

# add widgets to layouts and set layouts for windows
layout10_faces.addWidget(l_label6_faces)
layout10_faces.addWidget(l_faces_recognized)
window10_faces.setLayout(layout10_faces)

layout9_faces.addWidget(l_label5_faces)
layout9_faces.addWidget(l_face_recognition_time)
window9_faces.setLayout(layout9_faces)

layout8_faces.addWidget(l_label4_faces)
layout8_faces.addWidget(l_faces_detected)
window8_faces.setLayout(layout8_faces)

layout7_faces.addWidget(l_label3_faces)
layout7_faces.addWidget(l_face_detection_time)
window7_faces.setLayout(layout7_faces)

layout6_faces.addWidget(cb_recognition_method)
layout6_faces.addWidget(pb_recognize_faces)
window6_faces.setLayout(layout6_faces)

layout5_faces.addWidget(cb_face_detection_method)
layout5_faces.addWidget(pb_detect_faces)
window5_faces.setLayout(layout5_faces)

layout4_faces.addWidget(window7_faces)
layout4_faces.addWidget(window8_faces)
layout4_faces.addWidget(window9_faces)
layout4_faces.addWidget(window10_faces)
layout4_faces.addWidget(list_recognized_faces)
layout4_faces.addStretch()
window4_faces.setLayout(layout4_faces)

layout3_faces.addWidget(l_uploaded_image_faces)
layout3_faces.addWidget(pb_upload_image)
layout3_faces.addStretch()
window3_faces.setLayout(layout3_faces)

layout2_faces.addWidget(l_label1_faces)
layout2_faces.addWidget(window5_faces)
layout2_faces.addWidget(l_label2_faces)
layout2_faces.addWidget(window6_faces)
layout2_faces.addStretch()
window2_faces.setLayout(layout2_faces)

layout1_faces.addStretch()
layout1_faces.addWidget(window2_faces)
layout1_faces.addWidget(window3_faces)
layout1_faces.addWidget(window4_faces)
layout1_faces.addStretch()
window1_faces.setLayout(layout1_faces)

layout_faces.addWidget(l_title_faces)
layout_faces.addStretch()
layout_faces.addWidget(window1_faces)
layout_faces.addStretch()
layout_faces.addWidget(pb_quit)
window_faces.setLayout(layout_faces)
###########FACES TAB UI end############

###########OBJECTS TAB UI start############
# define widgets
l_title_objects = QtWidgets.QLabel("OBJECTS")
l_title_objects.setFont(QtGui.QFont("Courier", 38, QtGui.QFont.Bold))

l_label1_objects = QtWidgets.QLabel("detection method:")
l_label2_objects = QtWidgets.QLabel("detection time:")
l_label3_objects = QtWidgets.QLabel("objects detected:")
l_object_detection_time = QtWidgets.QLabel("0.00000")
l_objects_detected = QtWidgets.QLabel("0")
l_uploaded_image_objects = QtWidgets.QLabel("")
display_image_objects("white600x600.jpg")

pb_detect_objects = QtWidgets.QPushButton("Detect Objects")
pb_detect_objects.clicked.connect(pb_detect_objects_clicked)
pb_detect_objects.setDisabled(True)

cb_object_detection_method = QtWidgets.QComboBox()
cb_object_detection_method.addItem("Faster-RCNN (OpenImages)")
cb_object_detection_method.addItem("Faster-RCNN (Coco)")
cb_object_detection_method.addItem("Mask-RCNN (Coco)")
cb_object_detection_method.addItem("SSD Inception (Coco)")
cb_object_detection_method.addItem("SSD Resnet50 (Coco)")
cb_object_detection_method.addItem("YOLOv3 (OpenImages)")
cb_object_detection_method.addItem("YOLOv3 (Coco)")

list_detected_objects = QtWidgets.QListWidget()

# define windows
window_objects = QtWidgets.QWidget()
window1_objects = QtWidgets.QWidget()
window2_objects = QtWidgets.QWidget()
window3_objects = QtWidgets.QWidget()
window4_objects = QtWidgets.QWidget()
window5_objects = QtWidgets.QWidget()
window6_objects = QtWidgets.QWidget()
window7_objects = QtWidgets.QWidget()
window8_objects = QtWidgets.QWidget()
window9_objects = QtWidgets.QWidget()
window10_objects = QtWidgets.QWidget()

# define layouts
layout_objects = QtWidgets.QVBoxLayout()
layout1_objects = QtWidgets.QHBoxLayout()
layout2_objects = QtWidgets.QVBoxLayout()
layout3_objects = QtWidgets.QHBoxLayout()
layout4_objects = QtWidgets.QVBoxLayout()
layout5_objects = QtWidgets.QHBoxLayout()
layout6_objects = QtWidgets.QHBoxLayout()
layout7_objects = QtWidgets.QHBoxLayout()
layout8_objects = QtWidgets.QHBoxLayout()
layout9_objects = QtWidgets.QHBoxLayout()
layout10_objects = QtWidgets.QHBoxLayout()

# add widgets to layouts and set layouts for windows
layout7_objects.addWidget(l_label3_objects)
layout7_objects.addWidget(l_objects_detected)
window7_objects.setLayout(layout7_objects)

layout6_objects.addWidget(l_label2_objects)
layout6_objects.addWidget(l_object_detection_time)
window6_objects.setLayout(layout6_objects)

layout5_objects.addWidget(cb_object_detection_method)
layout5_objects.addWidget(pb_detect_objects)
window5_objects.setLayout(layout5_objects)

layout4_objects.addWidget(window6_objects)
layout4_objects.addWidget(window7_objects)
layout4_objects.addWidget(list_detected_objects)
layout4_objects.addStretch()
window4_objects.setLayout(layout4_objects)

layout3_objects.addWidget(l_uploaded_image_objects)
window3_objects.setLayout(layout3_objects)

layout2_objects.addWidget(l_label1_objects)
layout2_objects.addWidget(window5_objects)
layout2_objects.addStretch()
window2_objects.setLayout(layout2_objects)

layout1_objects.addStretch()
layout1_objects.addWidget(window2_objects)
layout1_objects.addWidget(window3_objects)
layout1_objects.addWidget(window4_objects)
layout1_objects.addStretch()
window1_objects.setLayout(layout1_objects)

layout_objects.addWidget(l_title_objects)
layout_objects.addWidget(window1_objects)
layout_objects.addStretch()
window_objects.setLayout(layout_objects)
###########OBJECTS TAB UI end############

layout_tab = QtWidgets.QVBoxLayout()
tab = QtWidgets.QTabWidget()
tab.setWindowTitle("Demo Program")
tab.addTab(window_faces, "faces")
tab.addTab(window_objects, "objects")
tab.resize(tab.minimumSizeHint())
tab.setMinimumSize(tab.minimumSizeHint())
tab.show()
sys.exit(app.exec_())
