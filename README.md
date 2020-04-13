# ArchivePrototype

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
