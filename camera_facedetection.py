import cv2
from time import time
from tools.control import timer_manager
from tools.drawing import annotate_objects
from tools.processing import *
from tools.detection import *

scale = 0.75
cam = cv2.VideoCapture(0)  # Define camera

# %% Camera loop
scale_camera = 0.9
scale_classifier = 0.4
scale_facial = 0.5
while (True):
# with timer_manager():
    start_time = time()

    # with timer_manager("FrameGrab"):
    img = resize(grab_frame(cam), scale_camera)

    # with timer_manager("Detection"):
    gs_img = resize(convert_to_gs(img), scale_classifier)

    faces = detect_object(gs_img, 'face_front', img)
    if len(faces) != 0:
        annotate_objects(img, faces, 'Face (frontal)', 1)
        for i, (x,y,w,h) in enumerate(faces):
            face_crop = img[y:y+h+20, x:x+w]
            # cv2.imshow('Face {}'.format(i), face_crop)
            eyes = detect_object(resize(face_crop, scale_facial), 'eye', face_crop, minNeighbors=40)
            mouths = detect_object(resize(face_crop, scale_facial), 'mouth', face_crop, minNeighbors=160)
            eyes = [(x_+x, y_+y, w, h) for (x_, y_, w, h) in eyes]
            if len(eyes) != 0:
                annotate_objects(img, eyes, 'Eye', 2)
            mouths = [(x_ + x, y_ + y, w, h) for (x_, y_, w, h) in mouths]
            if len(mouths) != 0:
                annotate_objects(img, mouths, 'Smile', 3)

    # with timer_manager("Draw"):
    end_time = time()
    fps = 1/(end_time-start_time)
    cv2.rectangle(img, (0, 0), (150, 40), (0, 0, 0), -1)
    annotation_str = "FPS: {:.2f}".format(fps)
    cv2.putText(img, annotation_str, (10, 30),
        fontFace=cv2.QT_FONT_NORMAL, fontScale=0.75, color=(255,255,255))
    cv2.imshow('Image', img)

    # cv2.imshow('frame', edges_horizontal)q;q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
