import cv2
import matplotlib.pyplot as plt
import numpy as np

def annotate_objects(img, objects, object_str, color_idx=1):
    color = plt.get_cmap('hsv', 5)(color_idx)
    color = (np.array(color)*255)[:-1]
    color = [int(c) for c in color]
    print("\t{} detected: {}".format(object_str, len(objects)))
    for i, (x, y, w, h) in enumerate(objects):
        annotation_str = "{} #{}".format(
            object_str, i + 1)
        x_pos = np.clip(x+20, 0, img.shape[1])
        y_pos = np.clip(y+30, 0, img.shape[0])
        cv2.putText(img, annotation_str, (x_pos, y_pos),
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=0.75, color=color)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
