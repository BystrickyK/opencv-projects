import numpy as np
import cv2
from scipy import ndimage as sig
from scipy import signal as s
from tools.filters import *
from tools.control import timer_manager
from tools.drawing import annotate_objects
import matplotlib.pyplot as plt
import matplotlib as mpl
import cupy as cp

scale = 0.75
def grab_frame(cam):
    ret, frame = cam.read()
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame
    return img

def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_to_gs(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



def resize(img, scale):
    w, h = img.shape[0], img.shape[1]
    w_after = int(w*scale)
    h_after = int(h*scale)
    img_after = cv2.resize(img, (h_after, w_after), interpolation=cv2.INTER_AREA)
    return img_after


cam = cv2.VideoCapture(0)  # Define camera

frontal_face_classifier = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
# profile_face_classifier = cv2.CascadeClassifier('classifiers/haarcascade_profileface.xml')
eye_classifier = cv2.CascadeClassifier('classifiers/haarcascade_eye.xml')
mouth_classifier = cv2.CascadeClassifier('classifiers/haarcascade_smile.xml')

def detect_object(gs_img, classifier, img, minNeighbors=20):
    scale = gs_img.shape[0] / img.shape[0]
    objects = classifier.detectMultiScale(gs_img, minNeighbors=minNeighbors)
    objects = list((np.array(objects) / scale).astype('uint32'))
    return objects


# %% Camera loop
scale_camera = 0.9
scale_classifier = 0.4
scale_facial = 0.5
while (True):
    with timer_manager():

        # with timer_manager("FrameGrab"):
        img = resize(grab_frame(cam), scale_camera)

        with timer_manager("Detection"):
            gs_img = resize(convert_to_gs(img), scale_classifier)

            faces = detect_object(gs_img, frontal_face_classifier, img)
            if len(faces) is not 0:
                annotate_objects(img, faces, 'Face (frontal)', 1)
                for i, (x,y,w,h) in enumerate(faces):
                    face_crop = img[y:y+h+20, x:x+w]
                    # cv2.imshow('Face {}'.format(i), face_crop)
                    eyes = detect_object(resize(face_crop, scale_facial), eye_classifier, face_crop, minNeighbors=40)
                    eyes = [(x_+x, y_+y, w, h) for (x_, y_, w, h) in eyes]
                    if len(eyes) is not 0:
                        annotate_objects(img, eyes, 'Eye', 2)
                    mouths = detect_object(resize(face_crop, scale_facial), mouth_classifier, face_crop, minNeighbors=160)
                    mouths = [(x_ + x, y_ + y, w, h) for (x_, y_, w, h) in mouths]
                    if len(mouths) is not 0:
                        annotate_objects(img, mouths, 'Smile', 3)

        # with timer_manager("Draw"):
        cv2.imshow('Image', img)

    # cv2.imshow('frame', edges_horizontal)q;q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()