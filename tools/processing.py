import cv2

def grab_frame(cam, channel=None):
    ret, frame = cam.read()
    if channel is None:
        return frame
    else:
        return frame[:, :, channel]

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
