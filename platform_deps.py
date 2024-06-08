"""
Determine what kind of platform we're running on,
and other platform-aware functions.
"""
import platform
import logging
import cv2


def is_windows():
    return platform.system() == 'Windows'


def is_v_10():
    return platform.version().startswith('10L')


def is_java():
    return platform.system() == "Java"


def is_linux():
    return platform.system() == "Linux"


def open_camera(index=0, ):
    """
    For platform-dependent params, etc.
    :param index: camera index
    :returns:  VideoCapture object
    """
    if is_windows():
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        logging.info("Camera opened for Windows OS.")
    else:
        cam = cv2.VideoCapture(index)
        logging.info("Camera opened for non-windows OS.")

    return cam
