# ArchivePrototype

## Project Structure
![project structure](./images/image(9).png "Project Structure")


### Prototype folder: 

this folder is the main folder and contains all the subfolders and the five python files with the application code.
It also contains a ‘.jpg’ file named ‘white600x600’ which is used as background image for the image upload area in the UI.
* Prototyp.py
* FaceDetect.py 
* FaceRecog.py 
* ObjectDetect.py
* ObjectLists.py
* white600x600.jpg


### Face recognition folder: 

the KNN classifier (‘trained_knn_model.clf’) used for face recognition is located in the ‘face recognition’ folder. 
The image sets that are used for training the KNN classier are organized in subfolders (‘F.W de Klerk’, ‘N. Mandela’) 
of the folder ‘knn train’. They currently consist of 17 ‘.jpg’ files per person.
*  trained_knn_model.clf (KNN Classifier)
*  F.W de Klerk (Class 1)
*  N. Mandela (Class 2)


### Face detection models folder: 

this folder contains the models used for face detection. Dlib’s CNN-Based Face Detector and Dlib’s HOG-Based Face Detector come with the library 
and therefore don’t have to be stored here.
* haarcascade_frontalface_default.xml (OpenCV’s Haar Cascade Face Detector)
* TFConfig.pbtxt, TFmodel.pb (OpenCV’s TensorFlow model architecture and weights)
* caffeDeploy.prototxt, res10_300x300_ssd_iter_140000.caffemodel (OpenCV’s Caffe model architecture and weights) 


### Object detection models folder: 

the TensorFlow object detection models as well as the YOLOv3-608 models are stored here. **You must download the models from the provided links and name them accordingly**.
* faster_rcnn_oi.pb (See [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md): Faster R-CNN Inception ResNet v2 Atrous Oid v4) 
* faster_rcnn_coco.pb (See [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md): Faster R-CNN Nas COCO) 
* mask_rcnn_coco.pb (See [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md): Mask R-CNN ResNet50 Atrous COCO) 
* ssd_inception_coco.pb (See [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md): SSD Inception v2 COCO) 
* ssd_resnet50_coco.pb (See [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md): SSD ResNet50 v1 FPN Shared Box Predictor 640x640 COCO14 Sync)
* yolov3oi.cfg, yolov3oi.weights (See [here](https://pjreddie.com/darknet/yolo/): the YOLOv3-608 model that was trained on the Open Images dataset  plus weights)
* yolov3.cfg, yolov3.weights (See [here](https://pjreddie.com/darknet/yolo/): the YOLOv3-608 model that was trained on the COCO dataset  plus weights) 

### Temp files folder: 

the temp files folder contains files that are generated during use of the software. 
These are the normalized picture and the pictures with the drawn in bounding boxes, 
which are displayed in the UI in the image upload area.
* normalizeTemp.jpg
* temp.jpg
* tempObjects.jpg
* tempRecog.jpg


### Installed packages
* numpy 1.17.0
* opencv-python 4.2.0.32
* scikit-learn 0.21.3
* tensorflow 2.1.0
* Pillow 6.0.0
* PySide2 5.12.3
* face-recognition 1.2.3


## User Interface
The UI is a window with two tabs. The ‘objects’-  and the ‘faces’ tab. 
The ‘faces’ tab contains everything related to face detection and face recognition, 
whereas the ‘objects’ tab contains everything related to object detection.

![Objects Tab](./images/image(17).png "Objects Tab")

The workflow is as follows. First an image has to be uploaded in the faces tab. 
Otherwise none of the other functionality will be available. By clicking the ‘Upload Image …’ button, a file dialog opens 
and a ‘.jpg’ image of any size and color can be selected by the user. 
The uploaded file is normalized automatically and displayed in the center of the window. 
If no image has been uploaded yet, a white square is displayed in the center instead. 
After the image was uploaded, the user can detect objects by selecting one of the seven object detection methods 
from the drop-down menu and clicking the button ‘Objects’ in the ‘objects’ tab. 
Red bounding boxes with the class names will be put around the detected objects and the detected classes. 
The confidence they were detected with will be displayed in a small window next to the image. 
It’s also displayed how long it took to detect the objects and how many were detected. 
Some of the models take a little longer to detect objects. 
So sometimes it is necessary to wait a little bit until results are visible.

![Faces Tab](./images/image(16).png "Faces Tab")

Alternatively, the user can detect faces by selecting one of the seven detection models from the drop-down menu 
in the ‘faces’ tab and clicking the button ‘Detect Faces’. Red bounding boxes will be drawn around the detected faces. 
The time it took the model to detect the faces, as well as the number of faces that were detected will be displayed next to the image. 
Only after the face detection took place, the user can recognize faces. 
In order to start the recognition, the user has to choose one choose one of the KNN options 
from the drop-down menu in the faces tab and click the ‘Recognize Faces’ button. 
The bounding boxes of the faces that were successfully recognized will turn blue 
and the name of the recognized person will appear beneath it. 
The bounding boxes of faces that could not be identified will stay red and the name ‘unknown’ will appear beneath it. 
A small window next to the image then shows the names of the recognized persons and the confidence they were recognized with. 
The number of faces that were recognized and the recognition time will be displayed as well. 
The user can upload a new picture or select other models at any time. 
As long as no detection or recognition is currently in progress. 
It can happen that with some models it takes a little longer to see results. 
The KNN is also affected. It is currently being trained anew with every execution.
