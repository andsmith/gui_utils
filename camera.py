"""
Simple cv2 wrapper, using callback for incoming frames.
"""
import cv2
import time
import numpy as np
from threading import Thread
import logging
import os


class Camera(object):

    def __init__(self, cam_ind, callback):
        """
        Acquire and start a webcam.
        :param cam_ind:  cv2 camera index
        :param callback:  function (frame, time) to call with new frames
        """
        self._cam_ind = cam_ind
        logging.info("Camera:  acquiring device %i..." % (cam_ind,))
        self._shutdown = False
        self._is_windows = os.name == 'nt'
        self._cam_thread = Thread(target=self._cam_thread_proc)
        self._callback = callback
        self._cam_thread.start()

    def shutdown(self):
        print("Camera got shutdown signal")
        self._shutdown = True

    def _cam_thread_proc(self, ):
        logging.info("Camera thread starting.")
        if self._is_windows:
            cam = cv2.VideoCapture(self._cam_ind, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(self._cam_ind)
        while not self._shutdown:
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


class CamTester(object):
    """Open a camera and show the video stream."""
    def __init__(self):
        self._cam = Camera(0, self._show_img)

    def _show_img(self, img, t):
        cv2.imshow("Any key to quit...", img)
        k = cv2.waitKey(1)
        if k != -1:
            self._cam.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CamTester()
