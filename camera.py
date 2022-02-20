"""
Simple cv2 wrapper, using callback for incoming frames.
"""
import cv2
import time
import numpy as np
from threading import Thread, Lock, Event

import logging
import os
from queue import Queue
from camera_settings import user_pick_resolution
from copy import deepcopy
import asyncio


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
        :param callback:  function (frame, time) to call with new frames
        :param settings: dict with params for VideoCapture.set(key=value)
            Settings are remembered by this class for each camera, so if it is switched back
            to a previously used one, it will resume those settings.

        :param prompt_resolution:  Ask user for camera resolution,

        """
        self._prompt_resolution = prompt_resolution
        self._cam_ind = cam_ind
        self._new_cam_ind = None  # to switch cameras
        self._no_network = False  # future?
        self._shutdown = False
        self._is_windows = os.name == 'nt'
        self._cam_thread = Thread(target=self._cam_thread_proc)
        self._callback = callback

        self._n_cameras = None  # don't count unless user asks to change cameras (save time)

        self._orig_settings = settings if settings is not None else {}
        self._settings = {self._cam_ind: deepcopy(self._orig_settings)}
        self._settings_lock = Lock()
        self._settings_changes_q = Queue()  # each should be a dict with one setting: value pair
        self._cam_thread.start()

    def shutdown(self):
        logging.info("Camera got shutdown signal")
        self._shutdown = True

    def change_cameras(self, cam_ind):
        if cam_ind == self._cam_ind:
            logging.info("Can't change to camera already in use.")
            return
        logging.info("Scheduling change to camera %i." % (cam_ind,))
        self._new_cam_ind = cam_ind

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

    def _apply_settings(self, cam, all=False):
        """
        Apply settings to camera.  If 'all' is True, all camera settings are applied,
        else only the ones in settings_changes_q are applied.

        :param cam: VideoCapture object
        :param all:  Apply all settings first?  (useful for first time cam is opened
        :return:   dict(setting constants:  current camera values)
        """
        current_settings = {}
        things_to_set = deepcopy(self._settings[self._cam_ind]) if all else {}
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
            current_settings[setting] = new_value
        return current_settings

    def _open_camera(self):
        """
        Open current camera, apply settings, prompt user if necessary
        :return: VideoCapture() object
        """

        if self._cam_ind not in self._settings:
            logging.info("First time opening camera %i." % (self._cam_ind,))
            # opening this camera for the first time
            self._settings[self._cam_ind] = deepcopy(self._orig_settings)

        # if it's in the settings, use old resolution, else ask user if interactive
        if cv2.CAP_PROP_FRAME_WIDTH not in self._settings[self._cam_ind]:
            logging.info("Camera %i settings does not contain resolution." % (self._cam_ind,))
            if self._prompt_resolution:
                resolution = user_pick_resolution(self._cam_ind, no_network=self._no_network)
                if resolution is None:
                    self.shutdown()
                    logging.info("User exit.")
                    raise ShutdownException()
                self._settings[self._cam_ind][cv2.CAP_PROP_FRAME_WIDTH] = resolution[0]
                self._settings[self._cam_ind][cv2.CAP_PROP_FRAME_HEIGHT] = resolution[1]

        logging.info("Acquiring camera %i..." % (self._cam_ind,))
        if self._is_windows:
            cam = cv2.VideoCapture(self._cam_ind, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(self._cam_ind)
        logging.info("Camera %i acquired." % (self._cam_ind,))

        self._apply_settings(cam, all=True)
        return cam

    def _cam_thread_proc(self, ):
        try:
            cam = self._open_camera()
        except ShutdownException:
            return

        while not self._shutdown:

            if self._new_cam_ind is not None:
                cam.release()
                self._cam_ind = self._new_cam_ind
                try:
                    cam = self._open_camera()
                except ShutdownException:
                    return
                self._new_cam_ind = None

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
            frame = np.ascontiguousarray(frame[:, ::-1, :])  # mirror image, not real image
            self._callback(frame, frame_time)
        logging.info("Camera:  releasing device...")
        cam.release()
        logging.info("Camera thread finished.")

    @staticmethod
    def count_cameras():
        """
        See how many cameras are attached to the computer.
        :return: number of cameras successfully opened

        WARNING:  Will not behave predictably if any cameras are open
        """
        n = 0
        c = None
        while True:
            try:
                if os.name == 'nt':  # windows, to avoid warning
                    c = cv2.VideoCapture(n, cv2.CAP_DSHOW)
                else:
                    c = cv2.VideoCapture(n)
                ret, frame = c.read()
                if frame.size < 10:
                    raise Exception("Out of cameras!")
                c.release()
                cv2.destroyAllWindows()
                n += 1
            except:
                c.release()
                cv2.destroyAllWindows()
                break
        return n


class CamTester(object):
    """Open a camera and show the video stream."""

    def __init__(self, cam_index=0, settings=None):
        self._n_frames = 0
        self._print_interval = 30
        self._t_start = None
        self._cam_index = cam_index
        self._n_cams = Camera.count_cameras()
        logging.info("Computer has %i cameras." % (self._n_cams))
        self._cam = Camera(self._cam_index, self._show_img, settings=settings, prompt_resolution=False)

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
        if k==ord('k'):
            self._cam_index = (self._cam_index+1) % self._n_cams
            print("Changing to camera %i." % (self._cam_index,))
            self._cam.change_cameras(self._cam_index)
        elif k != -1:
            self._cam.shutdown()



def _test_camera():
    CamTester(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_camera()
