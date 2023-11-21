"""
Simple cv2 wrapper, using callback for incoming frames.
"""
import cv2
import time
import numpy as np
import logging
import os
from threading import Thread, Lock
from queue import Queue
from copy import deepcopy

from .camera_settings import user_pick_resolution, count_cameras
from .gui_picker import choose_item_text, ChooseItemDialog


def get_cv2_prop_names():
    props = [s for s in dir(cv2) if s.startswith("CAP_PROP_")]
    return {getattr(cv2, prop): prop for prop in props}


class ShutdownException(Exception):
    pass


class Camera(object):
    _PROPS = get_cv2_prop_names()
    WIDTH_FLAG = cv2.CAP_PROP_FRAME_WIDTH
    HEIGHT_FLAG = cv2.CAP_PROP_FRAME_HEIGHT

    def __init__(self, cam_ind, callback=None, settings=None, prompt_resolution=False, mirror=True):
        """
        Webcam wrapper.  If user is asked for resolution, don't return until then.

        :param cam_ind:  cv2 camera index
        :param callback:  function(frame, time) to call with new frames
        :param settings: dict with params for VideoCapture.set(key=value)

        :param prompt_resolution:  Ask user for camera resolution before starting.

        """
        self._prompt_resolution = prompt_resolution
        self._cam_ind = cam_ind
        self._shutdown = False
        self._mirror = mirror
        self._started = False
        self._is_windows = os.name == 'nt'
        self._cam_thread = Thread(target=self._cam_thread_proc)
        self._callback = callback

        self._settings = {}
        self._settings_lock = Lock()  # need to be set in same thread as camera
        self._settings_changes_q = Queue()  # each should be a dict with setting--value pairs
        if settings is not None:
            self._settings_changes_q.put(settings)
            logging.info("%i settings queued to be applied." % (len(settings),))
        self._resolution = None

        if prompt_resolution:
            resolution = user_pick_resolution(self._cam_ind)
            if resolution is None:
                self.shutdown()
                logging.info("User exit.")
                raise ShutdownException()
            width, height = resolution
            self._settings_changes_q.put({Camera.WIDTH_FLAG: width,
                                          Camera.HEIGHT_FLAG: height})

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

    def get_resolution(self, wait=False):
        while wait and self._resolution is None:
            logging.info("Waiting for camera to start...")
            time.sleep(0.2)
        return self._resolution

    def set_resolution(self, target_resolution=None):
        """
        Add settings changes to queue (should happen in camera thread to be safe).
        """
        if target_resolution is not None:
            width, height = target_resolution
            self._settings_changes_q.put({Camera.WIDTH_FLAG: width,
                                          Camera.HEIGHT_FLAG: height})
        else:
            logging.info("No target resolution, camera not changed.")

    def _apply_settings(self, cam):
        """
        Apply queued setting changes to camera.
        :param cam: VideoCapture object
        """
        things_to_set = {}
        while not self._settings_changes_q.empty():
            things_to_set.update(self._settings_changes_q.get(block=True))

        for setting in things_to_set:
            name = self._PROPS[setting]
            logging.info("Setting camera property '%s' (%i):  %s" % (name, setting, things_to_set[setting]))
            cam.set(setting, things_to_set[setting])

        for setting in things_to_set:
            new_value = cam.get(setting)
            name = self._PROPS[setting]
            logging.info("New camera property '%s' (%i):  %s" % (name, setting, new_value))

    def _open_camera(self):
        """
        Open current camera, apply settings, prompt user if necessary
        :return: VideoCapture() object
        """

        logging.info("Acquiring camera %i..." % (self._cam_ind,))
        if self._is_windows:
            cam = cv2.VideoCapture(self._cam_ind, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(self._cam_ind)
        logging.info("Camera %i acquired." % (self._cam_ind,))
        self._apply_settings(cam)

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
            if not self._settings_changes_q.empty():
                self._apply_settings(cam)

            # grab data & send to callback
            ret, frame = cam.read()
            frame_time = time.perf_counter()
            if not ret:
                logging.warning("Camera not getting data, sleeping for a bit...")
                time.sleep(.1)
                continue

            if self._mirror:
                frame = np.ascontiguousarray(frame[:, ::-1, :])
            else:
                frame = np.ascontiguousarray(frame)  # mirror image, not real image
            if self._callback is not None:
                self._callback(frame, frame_time)

        logging.info("Camera:  releasing device...")
        cam.release()
        logging.info("Camera thread finished.")


def pick_camera(gui=True):
    """
    Ask user which camera to use.
    """
    prompt = "Please select one of the detected cameras:"
    print("Detecting cameras...")
    n_cams = count_cameras()
    logging.info("Detected %i cameras." % (n_cams,))
    if n_cams < 1:
        raise Exception("Webcam required for this version")
    elif n_cams == 1:
        cam_ind = 0
    else:
        choices = ['camera 0 (for laptops, probably user-facing)',
                   'camera 1 (probably forward-facing)']
        choices.extend(["camera %i" % (ind + 2,) for ind in range(n_cams - 2)])
        if gui:
            chooser = ChooseItemDialog(prompt=prompt)
            cam_ind = chooser.ask_text(choices)
        else:
            cam_ind = choose_item_text(choices, prompt)
        if cam_ind is None:
            raise Exception("User selected no camera.")
        print("Chose", cam_ind)
    return cam_ind

