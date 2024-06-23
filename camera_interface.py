"""
Base class for object that generate a periodic sequence of images, e.g. a camera or a background frame generator.
"""
import logging
import time
import numpy as np
import cv2
from abc import ABCMeta, abstractmethod

from .camera_settings import UserSettingsManager
from queue import Queue
from threading import Lock


class CameraBase(object, metaclass=ABCMeta):

    def __init__(self, resolution_wh=None, window_name="Camera", callback=None, mouse_callback=None):
        """
        :param resolution_wh:  Tuple with (width, height), or None to ask user/use defaults.
        :param frame_callback:  function(frame, time) to call with new frames
        :param mouse_callback:  function(event, x, y, flags, param) to call with mouse events
        :param window_name:  Name of the window to display the camera feed in.
        """
        self._window_name = window_name
        self._resolution_wh = resolution_wh
        self._shutdown = False
        self._started = False
        self._frame_callback = callback
        self._mouse_callback = mouse_callback

    @abstractmethod
    def start(self):  # does not return
        pass

    @abstractmethod
    def shutdown(self):
        pass

    def set_frame_callback(self, new_callback=None):
        self._frame_callback = new_callback
        logging.info("Frame callback changed.")

    def set_mouse_callback(self, new_callback=None):
        self._mouse_callback = new_callback
        logging.info("Mouse callback set.")

    @abstractmethod
    def set_resolution(self, res_w_h=None):
        pass
