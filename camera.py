"""
Simple cv2 wrapper, using callback for incoming frames.
"""
import cv2
import time
import numpy as np
from threading import Thread
import logging
import os
from queue import Queue
from camera_settings import user_pick_resolution

class Camera(object):

    def __init__(self, cam_ind, callback, settings = None):
        """
        Acquire and start a webcam.
        :param cam_ind:  cv2 camera index
        :param callback:  function (frame, time) to call with new frames
        :param settings: list of dict with params for VideoCapture.set(key=value)
            should have keys for setting (cv.CAP_PROP...), name (e.g. "fps"), and value (e.g. 60)
        """
        self._cam_ind = cam_ind
        logging.info("Camera:  acquiring device %i..." % (cam_ind,))
        self._shutdown = False
        self._is_windows = os.name == 'nt'
        self._cam_thread = Thread(target=self._cam_thread_proc)
        self._callback = callback
        self._settings_changes_q = Queue()
        for setting in settings:
            self._settings_changes_q.put(setting)
        self._cam_thread.start()

    def shutdown(self):
        print("Camera got shutdown signal")
        self._shutdown = True

    def set_resolution(self, target_resolution=None):
        """
        Add settings changes to queue (should happen in camera thread to be safe).

       """
        if target_resolution is not None:
            width, height = target_resolution
            self._settings_changes_q.put({'setting': cv2.CAP_PROP_FRAME_WIDTH,
                                          'value': width,
                                          'name': "width"})
            self._settings_changes_q.put({'setting': cv2.CAP_PROP_FRAME_height,
                                          'value': height,
                                          'name': "height"})
            logging.info("Resolution change added to settings change queue:  %i x %i" % target_resolution)
        else:
            logging.info("No target resolution, camera not changed.")

    def _apply_settings(self, cam):
        """
        Try to apply all settings changes in the queue, return new values
        :param cam: VideoCapture object
        :return:  list of dict('setting': enum constant, 'name': description, 'value':  current camera value)
        """
        new_values = []
        while not self._settings_changes_q.empty():
            setting = self._settings_changes_q.get(block=True)
            logging.info("Setting camera property '%s' (%i):  %s" % (setting['name'],
                                                                     setting['setting'],
                                                                     setting['value']))
            cam.set(setting['setting'], setting['value'])
            new_values.append({'setting': setting['setting'],'name': setting['name']})
        for i in range(len(new_values)):
            new_values[i]['value'] = cam.get(new_values[i]['setting'])
            logging.info("New camera property '%s' (%i):  %s" % (new_values[i]['name'],
                                                                 new_values[i]['setting'],
                                                                 new_values[i]['value']))
        return new_values

    def _open_camera(self):

        logging.info("Acquiring camera %i..." % (self._cam_ind,))
        if self._is_windows:
            cam = cv2.VideoCapture(self._cam_ind, cv2.CAP_DSHOW)
        else:
            cam = cv2.VideoCapture(self._cam_ind)
        logging.info("Camera %i acquired." % (self._cam_ind,))

        return cam

    def _cam_thread_proc(self, ):
        cam = self._open_camera()
        while not self._shutdown:

            if not self._settings_changes_q.empty():
                self._apply_settings(cam)

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

    def __init__(self, cam_index=0, settings=None):
        self._n_frames = 0
        self._print_interval = 30
        self._t_start = None
        self._cam = Camera(cam_index, self._show_img, settings = settings)

    def _show_img(self, img, t):
        self._n_frames += 1
        now = time.perf_counter()
        if self._t_start is None:
            self._t_start = now
        elif now - self._t_start > 1.0:
            delta_t = now - self._t_start
            print("FPS:  %.3f, last frame:  %s" % (self._n_frames / delta_t, (img.shape[1], img.shape[0])))
            self._n_frames = 0
            self._t_start = now

        cv2.imshow("Any key to quit...", img)
        k = cv2.waitKey(1)
        if k != -1:
            self._cam.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    camera = 0
    res = user_pick_resolution(camera, gui=False)
    if res is None:
        print("Quit!")
    else:
        width, height = res
        settings = [{'setting': cv2.CAP_PROP_FRAME_WIDTH, 'name': 'width', 'value': width},
                    {'setting': cv2.CAP_PROP_FRAME_HEIGHT, 'name': 'height', 'value': height},]
        CamTester(camera, settings=settings)
