# ArchivePrototype

![project structure](./images/image(9).png "Project Structure")


### Prototype folder: 

this folder is the main folder and contains all the subfolders and the five python files with the application code.
It also contains a ‘.jpg’ file named ‘white600x600’ which is used as background image for the image upload area in the UI.
* ‘Prototyp.py’ 
* ‘FaceDetect.py’ 
* ‘FaceRecog.py’ 
* ‘ObjectDetect.py’ ‘
* ObjectLists.py’ 
* ‘white600x600.jpg’


### Face recognition folder: 

the KNN classifier (‘trained_knn_model.clf’) used for face recognition is located in the ‘face recognition’ folder. 
The image sets that are used for training the KNN classier are organized in subfolders (‘F.W de Klerk’, ‘N. Mandela’) 
of the folder ‘knn train’. They currently consist of 17 ‘.jpg’ files per person.
* ‘trained_knn_model.clf’
* ‘F.W de Klerk’
* ‘N. Mandela’


### Face detection models folder: 

this folder contains the models used for face detection. Dlib’s CNN-Based Face Detector and Dlib’s HOG-Based Face Detector come with the library 
and therefore don’t have to be stored here.
* OpenCV’s Haar Cascade Face Detector (‘haarcascade_frontalface_default.xml’), 
* OpenCV’s TensorFlow model architecture (‘TFConfig.pbtxt’) and weights (‘TFmodel.pb’) 
* OpenCV’s Caffe model architecture (‘caffeDeploy.prototxt’) and weights (‘res10_300x300_ssd_iter_140000.caffemodel’) 


### Object detection models folder: 

the TensorFlow object detection models as well as the YOLOv3-608 models are stored here. 
* ‘Faster R-CNN Inception ResNet v2 Atrous Oid v4’ (‘faster_rcnn_oi.pb’)
* ‘Faster R-CNN Nas COCO’ (‘faster_rcnn_coco.pb’) 
* ‘Mask R-CNN ResNet50 Atrous COCO’ (‘mask_rcnn_coco.pb’) 
* ‘SSD Inception v2 COCO’ (‘ssd_inception_coco.pb’)
* ‘SSD ResNet50 v1 FPN Shared Box Predictor 640x640 COCO14 Sync / RetinaNet’ (‘ssd_resnet50_coco.pb’)
* the YOLOv3-608 model that was trained on the Open Images dataset (‘yolov3oi.cfg’) plus weights (‘yolov3oi.weights’) 
* the YOLOv3-608 model that was trained on the COCO dataset (‘yolov3.cfg’) plus weights (‘yolov3.weights’).

### Temp files folder: 

the temp files folder contains files that are generated during use of the software. 
These are the normalized picture and the pictures with the drawn in bounding boxes, 
which are displayed in the UI in the image upload area.
* ‘normalizeTemp.jpg’ 
* ‘temp.jpg’, ‘tempObjects.jpg’
* ‘tempRecog.jpg’) 