import numpy as np
import cv2

classifiers = {
'face_front': cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml'),
'face_profile': cv2.CascadeClassifier('classifiers/haarcascade_profileface.xml'),
'eye': cv2.CascadeClassifier('classifiers/haarcascade_eye.xml'),
'mouth': cv2.CascadeClassifier('classifiers/haarcascade_smile.xml')
}

def detect_object(gs_img, classifier_str, img, minNeighbors=20):
    scale = gs_img.shape[0] / img.shape[0]
    classifier = classifiers[classifier_str]
    objects = classifier.detectMultiScale(gs_img, minNeighbors=minNeighbors)
    objects = list((np.array(objects) / scale).astype('uint32'))
    return objects
