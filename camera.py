"""
CV2 wrapper to manage detection of system camera capabilities and user's selection.
"""
import cv2
import time
import numpy as np
import logging
from threading import Lock
from queue import Queue
from copy import deepcopy
from .platform_deps import open_camera
from .camera_settings import UserSettingsManager
from .video_io_base import VideoBase


def get_cv2_prop_names():
    props = [s for s in dir(cv2) if s.startswith("CAP_PROP_")]
    return {getattr(cv2, prop): prop for prop in props}


class ShutdownException(Exception):
    pass


class Camera(VideoBase):
    """
    Creating a Camera object with no arguments will load the current configuration or use the default if none exists.
    To create a more specific camera object (e.g. which camera, what resolution), pass in the appropriately constructed
    UserSettingsManager object, which will scan/ask for any missing information required for the VideoCapture.
    """
    _PROPS = get_cv2_prop_names()
    WIDTH_FLAG = cv2.CAP_PROP_FRAME_WIDTH
    HEIGHT_FLAG = cv2.CAP_PROP_FRAME_HEIGHT

    def __init__(self, index=0, frame_res=None, disp_res=None, window_name="Camera",
                 callback=None, mouse_callback=None, keyboard_callback=None,
                 window_flags=cv2.WINDOW_AUTOSIZE, quiet=False, ask_user='gui', mirrored=True):
        """
        :param index:  Index of the camera to use, or None to ask user/use defaults.
        :param frame_res:  Tuple with (width, height), or None to ask user/use defaults.
        :param disp_res:  Tuple with (width, height), or None to use frame resolution.
        :param window_name:  Name of the window to display the camera feed in.
        :param callback:  function(frame, time) to call with new frames
        :param mouse_callback:  function(event, x, y, flags, param) to call with mouse events
        :param keyboard_callback:  function(key) to call with keyboard events
        :param window_flags:  OpenCV window flags
        :param quiet:  If True, don't print FPS info to the console.
        :param ask_user:  'gui' to prompt user for settings, 'cli' to use defaults, 'silent' to use defaults.
        :param mirrored:  True to mirror the image, False to leave it as-is.

        """
        # settings manager will figure out resolution if not provided
        self._disp_res = disp_res
        self._settings = UserSettingsManager(index=index,
                                             res_w_h=frame_res,
                                             mirrored=mirrored,
                                             interaction=ask_user)
        actual_frame_res = self._settings.get_resolution_wh()

        super().__init__(actual_frame_res, disp_res, window_name,
                         callback=callback, mouse_callback=mouse_callback,
                         keyboard_callback=keyboard_callback, window_flags=window_flags,
                         quiet=quiet)
        self._pending_settings = Queue()
        self._settings_lock = Lock()

    def _disambiguate_resolution(self, frame_res, disp_res):
        """
        This was already done by the UserSettingsManager, so just return the frame resolution it returned.
        """
        return self._settings.get_resolution_wh(), self._disp_res

    def _stop_making_frames(self):
        """
        Camera thread should be watching for self._stop to be True, and exit if it is, so there's no need to do anything here.
        """
        pass

    def change_settings(self, new_settings):
        """
        Keys are CV2 camera setting constants.

        :param new_settings:  dict(setting_1=value_1, ...)
        """
        with self._settings_lock:
            self._pending_settings.put(new_settings)
        logging.info("Queued settings change:  %s" % (new_settings,))

    def _set_camera_resolution(self, res_w_h=None):
        """
        Add settings changes to queue (should happen in camera's thread to be safe).
                (convenience wrapper for change_settings)
        :param res_w_h:  tuple(width, height)
        """
        width, height = res_w_h
        self.change_settings({Camera.WIDTH_FLAG: width,
                              Camera.HEIGHT_FLAG: height})
        logging.info("Queued change to resolution:  %i x %i " %
                     (width, height,))

    def _flush_changes(self, cam):
        """
        Apply queued setting changes to camera, will crash if not called from same thread that opened camera.  This
        function should be called before each frame grab to make sure camera settings are as up-to-date as possible.
        :param cam:  VideoCapture object opened.
        """

        # process them in order to get most up-to-date set of settings
        things_to_set = {}
        while not self._pending_settings.empty():
            things_to_set.update(self._pending_settings.get(block=True))

        # send to camera
        for setting in things_to_set:
            name = self._PROPS[setting]
            logging.info("Setting camera property '%s' (%i):  %s" %
                         (name, setting, things_to_set[setting]))
            cam.set(setting, things_to_set[setting])

        # See if they stuck
        for setting in things_to_set:
            new_value = cam.get(setting)
            name = self._PROPS[setting]
            logging.info("New camera property '%s' (%i):  %s" %
                         (name, setting, new_value))
            if new_value != things_to_set[setting]:
                logging.warn(
                    "Attempted to change %s to %s, but result was %s." % (setting, things_to_set[setting], new_value))

    def _open_camera(self):
        """
        Open current camera, apply settings, prompt user if necessary
        :return: VideoCapture() object
        """
        index = self._settings.get_index()
        resolution = self._settings.get_resolution_wh()
        self._set_camera_resolution(resolution)  # will be enqueued here...
        logging.info("Acquiring camera %i..." % (index,))
        cam = open_camera(index)
        self._flush_changes(cam)  # ... and then set here.
        fps = cam.get(cv2.CAP_PROP_FPS)
        logging.info("\tCamera started with FPS:  %s" % (fps,))

        return cam

    def _start_making_frames(self, ):
        """
        Does not return.
        """
        try:
            cam = self._open_camera()
        except ShutdownException:
            return

        while not self._stop:
            # need to change settings?
            if not self._pending_settings.empty():
                # need to do in same thread
                self._flush_changes(cam)

            # grab data & send to callback
            ret, frame = cam.read()
            frame_time = time.perf_counter()
            if not ret:
                logging.warning(
                    "Camera returning no data, sleeping for a bit...")
                time.sleep(.1)
                continue

            if self._settings.is_mirrored():
                frame = np.ascontiguousarray(frame[:, ::-1, :])
            else:
                # mirror image, not real image
                frame = np.ascontiguousarray(frame)
            if self._callback is not None:
                self._callback(frame, frame_time)

        logging.info("Camera:  releasing device...")
        cam.release()
        logging.info("Camera thread finished.")


def camera_tester():
    """
    Open a camera, open a display window, and display the camera feed in it.
    """
    cam = Camera(window_name="Camera tester - Press 'q' to quit.")

    cam.set_frame_callback(cam.auto_display_callback)
    cam.start()
    logging.info("Camera thread finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    camera_tester()
