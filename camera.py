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

    def __init__(self, cam_ind, callback, settings=None, prompt_resolution=False):
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
        self._started = False
        self._is_windows = os.name == 'nt'
        self._cam_thread = Thread(target=self._cam_thread_proc)
        self._callback = callback

        self._settings = settings if settings is not None else {}
        self._settings_lock = Lock()  # need to be set in same thread as camera
        self._settings_changes_q = Queue()  # each should be a dict with one setting--value pair
        self._resolution = None
        self._target_resolution = None
        if prompt_resolution:
            resolution = user_pick_resolution(self._cam_ind)
            if resolution is None:
                self.shutdown()
                logging.info("User exit.")
                raise ShutdownException()
            self._target_resolution = resolution

    def start(self):
        if self._started:
            raise Exception("Camera already started!")
        self._started = True
        self._cam_thread.start()

    def shutdown(self):
        logging.info("Camera got shutdown signal")
        self._shutdown = True

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
            self._settings_changes_q.put({cv2.CAP_PROP_FRAME_WIDTH: width})
            self._settings_changes_q.put({cv2.CAP_PROP_FRAME_HEIGHT: height})
            self._resolution = tuple(np.int64(target_resolution))  # not effective immediately

            logging.info("Resolution change added to settings change queue:  %i x %i" % target_resolution)
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

        if self._target_resolution is not None:
            self.set_resolution(self._target_resolution)
            self._apply_settings(cam)
        self._resolution = tuple(np.int64([cam.get(cv2.CAP_PROP_FRAME_WIDTH),
                                           cam.get(cv2.CAP_PROP_FRAME_HEIGHT)]))
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
            frame = np.ascontiguousarray(frame)  # mirror image, not real image
            self._callback(frame, frame_time)

        logging.info("Camera:  releasing device...")
        cam.release()
        logging.info("Camera thread finished.")


class CamTester(object):
    """Open a camera and show the video stream."""

    def __init__(self, cam_index=0, settings=None):
        self._n_frames = 0
        self._print_interval = 30
        self._t_start = None
        self._cam_index = cam_index
        self._cam = Camera(self._cam_index, self._show_img, settings=settings, prompt_resolution=True)
        self._cam.start()

    def _show_img(self, img, t):
        self._n_frames += 1
        now = time.perf_counter()
        if self._t_start is None:
            self._t_start = now
        elif now - self._t_start > 1.0:
            delta_t = now - self._t_start
            logging.info("FPS:  %.3f, last frame:  %s" % (self._n_frames / delta_t, (img.shape[1], img.shape[0])))
            self._n_frames = 0
            self._t_start = now

        cv2.imshow("Any key to quit...", img)
        k = cv2.waitKey(1)
        if k != -1:
            self._cam.shutdown()


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


def _test_camera():
    cam_ind = pick_camera(gui=False)
    cam_ind = pick_camera(gui=True)
    print("Done testing camera picker.")
    CamTester(cam_ind)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_camera()
