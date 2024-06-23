"""
Generate frames of a solid color, send them somewhere periodically.
(i.e. as a substitute for a Camera() object when not using a camera.)
"""

import numpy as np
import logging
from .colors import RGB_COLORS as COLORS
from threading import Thread
import time


class BackgroundFrameGenerator(object):
    """
    Generate frames of a solid color, for when not using a camera.
    """

    def __init__(self, size, bkg=None, dt=1./30, frame_callback=None):
        """
        :param size: (width, height) of the frame
        :param bkg: background color, as a 3-tuple of RGB values (0-255)
        :param dt: time between frames, in seconds
        :param mouse_callback: callback function for mouse events


        """
        self._size = size
        self._dt = dt
        self._stop = False
        self._frame_callback = frame_callback
        self._bkg_color = bkg if bkg is not None else COLORS['dark_dark_gray']
        self._frame = np.zeros(
            (self._size[1], self._size[0], 3), dtype=np.uint8)
        self._frame[:, :, :] = self._bkg_color

        self._thread = Thread(target=self._run)
        logging.info("Background frame generator initialized")


    def set_frame_callback(self, callback):
        self._frame_callback = callback

    def start(self):
        logging.info("Starting background frame generator")
        self._thread.start()

    def stop(self):
        logging.info("Stopping background frame generator")
        self._stop = True
        self._thread.join()
        logging.info("Background frame generator stopped")

    def _run(self):
        while not self._stop:
            time.sleep(self._dt)
            if self._frame_callback:
                print("Sending frame")
                frame = self._frame.copy()
                self._frame_callback(frame, time.perf_counter())
            else:
                print("Not sending frame")
