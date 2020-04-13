"""
Face Recognition
Sources: https://github.com/ageitgey/face_recognition/blob/master/examples/face_recognition_knn.py
         https://github.com/ageitgey/face_recognition
"""

import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition as fr
from face_recognition.face_recognition_cli import image_files_in_folder
import time
import FaceDetect
import numpy as np

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def run_knn(path, knn):
    # print("train_knn classifier")
    classifier = train_knn("face recognition/knn train", model_save_path="face recognition/trained_knn_model.clf",
                           n_neighbors=2)

    predictions, confidence_scores = predict_knn(path, knn, model_path="face recognition/trained_knn_model.clf")

    image = show_prediction_labels_on_image_knn(path, predictions, confidence_scores)
    return image


def train_knn(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    x = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = fr.load_image_file(img_path)
            face_bounding_boxes = fr.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                x.append(fr.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(x))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train_knn the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict_knn(x_img_path, k, knn_clf=None, model_path=None, distance_threshold=0.5):
    global recog_count, start, end

    recog_count = 0
    confidence_scores = []

    if not os.path.isfile(x_img_path) or os.path.splitext(x_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(x_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply k classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    image = fr.load_image_file(x_img_path)
    face_locations = FaceDetect.get_new_locations()

    # If no faces are found in the image, return an empty result.
    if len(face_locations) == 0:
        return []

    start = time.time()
    # Find encodings for faces in the test image
    faces_encodings = fr.face_encodings(image, known_face_locations=face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=k)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]
    name_prediction = knn_clf.predict(faces_encodings)

    # Predict classes and remove classifications that aren't within the threshold
    predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                   zip(name_prediction, face_locations, are_matches)]
    end = time.time()

    # Count recognized persons			
    recog_count = np.sum(are_matches)

    # calculate the confidence score
    confidence_scores = calculate_confidence_score_knn(face_locations, closest_distances, distance_threshold,
                                                       name_prediction, k)

    return predictions, confidence_scores


def show_prediction_labels_on_image_knn(img_path, predictions, confidence_scores):
    global recognized_faces

    recognized_faces = []
    draw_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(draw_image)

    i = 0
    for name, (top, right, bottom, left) in predictions:
        if name == "unknown":
            color = "red"
        else:
            color = "blue"
            # add to recognized faces list
            confidence_score = confidence_scores[i]
            recognized_faces.append("{} {:.2f}%".format(name, confidence_score))
        i = i + 1

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")
        draw.rectangle([left, top, right, bottom], outline=color)
        draw.text([left, bottom], name, fill=color)

    # Remove the drawing library from memory as per the Pillow docs
    del draw
    return draw_image


def calculate_confidence_score_knn(face_locations, closest_distances, distance_threshold, name_prediction, knn):

    confidence_scores = []

    # for every neighbor of a detection that is a match (applies to the distance_threshold)
    for detection in range(len(face_locations)):
        rounded_scores_in_percent = []
        if closest_distances[0][detection][0] <= distance_threshold:
            for neighbor in range(0, knn):

                # neighbor numbers 0-16 are F.W. De Klerk and 17-33 are Nelson Mandela. (run_knn was trained with 34
                # images, 16 for every person )
                neighbor_number = closest_distances[1][detection][neighbor]
                distance = closest_distances[0][detection][neighbor]

                # calculate the average of the neighbors that match the prediction
                if (name_prediction[detection] == "N. Mandela") & (neighbor_number >= 17):
                    score = 1 / (1 + distance)
                    rounded_score_in_percent = 100 * (round(score, 4))
                    # print("Mandela D: {}, N: {}, rsc %: {}".format(detection, neighbor, rounded_score_in_percent))
                    rounded_scores_in_percent.append(rounded_score_in_percent)

                elif (name_prediction[detection] == "F.W. De Klerk") & (neighbor_number < 17):
                    score = 1 / (1 + distance)
                    rounded_score_in_percent = 100 * (round(score, 4))
                    # print("De Klerk D: {}, N: {}, rsc %: {}".format(detection, neighbor, rounded_score_in_percent))
                    rounded_scores_in_percent.append(rounded_score_in_percent)

            # calculate the average
            confidence_score = 0
            asc_sum = 0
            x = 0
            for i in rounded_scores_in_percent:
                asc_sum = asc_sum + i
                x = x + 1
            confidence_score = asc_sum / x
            confidence_scores.append(confidence_score)

        # if no match append an average score of 0
        else:
            confidence_scores.append(0)

    return confidence_scores


# get the time it took to recognize the faces
def get_recognition_time():
    return end - start


# get the number of faces recognized
def get_number_of_faces_recognized():
    return recog_count


# returns a list of all the faces that have been recognized
def get_recognized_faces():
    return recognized_faces
