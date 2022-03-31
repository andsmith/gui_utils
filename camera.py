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

from camera_settings import user_pick_resolution, count_cameras


def get_cv2_prop_names():
    props = [s for s in dir(cv2) if s.startswith("CAP_PROP_")]
    return {getattr(cv2, prop): prop for prop in props}


class ShutdownException(Exception):
    pass


class Camera(object):
    _PROPS = get_cv2_prop_names()

    def __init__(self, cam_ind, callback, settings=None, prompt_resolution=False):
        """
        Acquire and start a webcam.

        :param cam_ind:  cv2 camera index
        :param callback:  function(frame, time) to call with new frames
        :param settings: dict with params for VideoCapture.set(key=value)

        :param prompt_resolution:  Ask user for camera resolution before starting.

        """
        self._prompt_resolution = prompt_resolution
        self._cam_ind = cam_ind
        self._shutdown = False
        self._is_windows = os.name == 'nt'
        self._cam_thread = Thread(target=self._cam_thread_proc)
        self._callback = callback

        self._settings = settings if settings is not None else {}
        self._settings_lock = Lock()  # need to be set in same thread as camera
        self._settings_changes_q = Queue()  # each should be a dict with one setting--value pair
        self._cam_thread.start()

    def shutdown(self):
        logging.info("Camera got shutdown signal")
        self._shutdown = True

    def set_resolution(self, target_resolution=None):
        """
        Add settings changes to queue (should happen in camera thread to be safe).
       """
        if target_resolution is not None:
            width, height = target_resolution
            self._settings_changes_q.put({cv2.CAP_PROP_FRAME_WIDTH: width})
            self._settings_changes_q.put({cv2.CAP_PROP_FRAME_HEIGHT: height})
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
        target_resolution = None
        if self._prompt_resolution:
            resolution = user_pick_resolution(self._cam_ind)
            if resolution is None:
                self.shutdown()
                logging.info("User exit.")
                raise ShutdownException()
            target_resolution = resolution

        logging.info("Acquiring camera %i..." % (self._cam_ind,))
        if self._is_windows:
            cam = cv2.VideoCapture(self._cam_ind, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(self._cam_ind)
        logging.info("Camera %i acquired." % (self._cam_ind,))
        if target_resolution is not None:
            self.set_resolution(target_resolution)

        self._apply_settings(cam)
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
        self._n_cams = count_cameras()
        logging.info("Computer has %i cameras." % (self._n_cams))
        self._cam = Camera(self._cam_index, self._show_img, settings=settings, prompt_resolution=True)

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



def _test_camera():
    CamTester(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_camera()
