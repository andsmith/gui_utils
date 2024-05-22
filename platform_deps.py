import cv2
import platform


def open_camera(index=0):
    if platform.system() == 'Windows':
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cam = cv2.VideoCapture(index)
    return cam
