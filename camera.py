"""
CV2 wrapper to manage detection of system camera capabilities and user's selection.
"""
import cv2
import time
import numpy as np
import logging
import appdirs
import os
from enum import IntEnum
from threading import Thread, Lock
from queue import Queue
from copy import deepcopy
from .platform_deps import open_camera
from .camera_settings import UserSettingsManager


def get_cv2_prop_names():
    props = [s for s in dir(cv2) if s.startswith("CAP_PROP_")]
    return {getattr(cv2, prop): prop for prop in props}


class ShutdownException(Exception):
    pass


class Camera(object):
    """
    Creating a Camera object with no arguments will load the current configuration or use the default if none exists.
    To create a more specific camera object (e.g. which camera, what resolution), pass in the appropriately constructed
        UserSettingsManager object, which will scan/ask for any missing information required for the VideoCapture.
    """
    _PROPS = get_cv2_prop_names()
    WIDTH_FLAG = cv2.CAP_PROP_FRAME_WIDTH
    HEIGHT_FLAG = cv2.CAP_PROP_FRAME_HEIGHT

    def __init__(self, index=None, callback=None, ask_user='gui', mirrored=True, resolution_wh=None):
        """
        :param index:  cv2 camera index, or, if None then use value in user settings file (or default if index isn't
            set).
        :param callback:  function(frame, time) to call with new frames
        :param ask_user: If no camera settings file is found, or if it is missing information:
            'gui':  as user in a dialog box
            'console':  ask user w/text
            'quiet':  use default values for missing settings
        :param mirrored:  if True, images are horizontally flipped
        :param resolution_wh:  Tuple with (width, height), or None to ask user/use defaults.

        """
        self._settings = UserSettingsManager(index=index, res_w_h=resolution_wh, mirrored=mirrored, use_gui=ask_user)

        self._shutdown = False
        self._started = False
        self._callback = callback

        self._cam_thread = Thread(target=self._cam_thread_proc)

    def start(self):
        if self._started:
            raise Exception("Camera already started!")
        self._started = True
        self._cam_thread.start()

    def shutdown(self):
        logging.info("Camera got shutdown signal")
        self._shutdown = True

    def set_callback(self, new_callback=None):
        self._callback = new_callback

    def set_resolution(self, target_resolution=None):
        """
        Add settings changes to queue (should happen in camera's thread to be safe).
        """
        width, height = target_resolution
        self._settings.enqueue({Camera.WIDTH_FLAG: width,
                                Camera.HEIGHT_FLAG: height})

    def _open_camera(self):
        """
        Open current camera, apply settings, prompt user if necessary
        :return: VideoCapture() object
        """
        index = self._settings.get_index()
        logging.info("Acquiring camera %i..." % (index,))
        cam = open_camera(index)
        self._settings.flush_changes(cam)
        fps = cam.get(cv2.CAP_PROP_FPS)
        logging.info("\tDevice FPS:  %s" % (fps,))

        return cam

    def _cam_thread_proc(self, ):
        try:
            cam = self._open_camera()
        except ShutdownException:
            return

        while not self._shutdown:
            # need to change settings?
            if self._settings.changes_pending():
                # need to do in same thread
                self._settings.flush_changes(cam)

            # grab data & send to callback
            ret, frame = cam.read()
            frame_time = time.perf_counter()
            if not ret:
                logging.warning("Camera returning no data, sleeping for a bit...")
                time.sleep(.1)
                continue

            if self._settings.is_mirrored():
                frame = np.ascontiguousarray(frame[:, ::-1, :])
            else:
                frame = np.ascontiguousarray(frame)  # mirror image, not real image
            if self._callback is not None:
                self._callback(frame, frame_time)

        logging.info("Camera:  releasing device...")
        cam.release()
        logging.info("Camera thread finished.")
