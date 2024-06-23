"""
Base class for pipeline that generates images and displays to screen.
"""
import logging
import time
import numpy as np
import cv2
from abc import ABCMeta, abstractmethod

from .camera_settings import UserSettingsManager
from queue import Queue
from threading import Lock


class VideoBase(object, metaclass=ABCMeta):
    """
    Base class for video input/output.
        * Capture / create frames and send them somewhere (using a callback)
        * Take frames from caller(s) and display them on the screen.
        * Handle mouse and keyboard events with callbacks.
    (input and output are necessarily together  because in OpenCV they need to be in the same thread to not crash)
    """

    def __init__(self, frame_res=None, disp_res=None, window_name="Video Base",
                 callback=None, mouse_callback=None, keyboard_callback=None,
                 window_flags=cv2.WINDOW_AUTOSIZE, quiet=False):
        """
        :param frame_res:  Tuple with (width, height), resolution of camera, or of frames to generate, or None for default behavior..
        :param disp_res:  Tuple with (width, height), resolution of window to display frames in, or None to use res_in / default.
            NOTE: any shaped image will display in any shaped window, but the window will only change shape if set_output_resolution is called.
            (default behavior will be to change the window shape to the first frame's shape.)
        :param window_name:  Name of the display window.
        :param callback:  function(frame, time) to call with new frames
        :param keyboard_callback:  function(key) to call with keyboard events
        :param mouse_callback:  function(event, x, y, flags, param) to call with mouse events
        :param window_flags:  OpenCV window flags
        :param quiet:  If True, don't print FPS info to the console.
        """
        self._quiet = quiet
        self._window_name = window_name
        self._frame_res, self._disp_res = frame_res, disp_res
        self._stop = False  # loops should watch this and exit if it's True
        self._started = False
        self._callback = callback
        self._mouse_callback = mouse_callback
        self._keyboard_callback = keyboard_callback
        self._flags = window_flags

        self._fps_info = {
            'reporting_cycle_seconds': 2.,  # report FPS every 'reporting_cycle' frames
            't_start': time.perf_counter(),  # start time of the current reporting cycle
            'n_frames': 0,  # number of frames displayed in the current reporting cycle
            'fps': 0,  # avg frames per second in the last reporting cycle
            # total time spent idle (e.g. generating frames) in the last reporting cycle
            't_idle': 0.,
            # 'mean_idle_time': 0., # mean idle time per frame in the last reporting cycle
            't_busy': 0.,  # total time spent displaying frames in the last reporting cycle
            # 'mean_busy_time': 0.} # mean display time per frame in the last reporting cycle
        }
        self._t_display_complete = time.perf_counter()

    @abstractmethod
    def _start_making_frames(self):
        """
        Video-in.
        If using a real camera, start sending frames to the callback.
        If using a generator, start generating frames and sending them to the callback, etc.

        This function should not return until self._stop is True.
        """
        pass

    @abstractmethod
    def _stop_making_frames(self):
        """
        Video-in.  stop whatever is generating frames.  (i.e. a camera)
        """
        pass

    def start(self):
        if self._stop:
            raise Exception("Can't restart after shutdown!")
        logging.info("Video IO started.")
        self._start_making_frames()  # start the frame generator or camera
        # does not return until shutdown

    def shutdown(self):
        self._stop = True
        cv2.destroyWindow(self._window_name)
        self._stop_making_frames()
        logging.info("Video IO stopped.")

    @abstractmethod
    def _set_camera_resolution(self):
        """
        Interface with hardware if a real camera, else probably not much.
        """
        pass

    def set_frame_resolution(self, res=None):
        # Will need to override this in subclasses to actually change the resolution of a real camera.
        self._frame_res = res
        self._set_camera_resolution()
        logging.info("Image resolution set to %s" % (res,))

    def set_disp_resolution(self, res=None):
        self._disp_res = res
        cv2.resizeWindow(self._window_name, res[0], res[1])
        logging.info("Output resolution set to %s" % (res,))

    def set_frame_callback(self, new_callback=None):
        self._frame_callback = new_callback
        logging.info("Frame callback changed.")

    def set_mouse_callback(self, new_callback=None):
        self._mouse_callback = new_callback
        logging.info("Mouse callback set.")

    def set_keyboard_callback(self, callback=None):
        self._keyboard_callback = callback
        logging.info("Keyboard callback set.")

    def _init_display(self, input_resolution):
        """"
        Initialize the OpenCV window, w/the appropriate flags.
        If the display resolution is still not set set it to the input resolution.
        """
        cv2.namedWindow(self._window_name, self._flags)
        logging.info("Opened display window with flag:  %s" % (self._flags,))
        if self._disp_res is None:
            self._disp_res = input_resolution
            logging.info("Display resolution set to %s" % (input_resolution,))
        # self.set_disp_resolution(self._disp_res)  # TODO: FIX THIS, figure out why uncommenting slows down the display

        self._started = True

    def show(self, frame):
        """
        Show a frame and handle keyboard input. NOTE: this must be called from the same thread as the callback that received the new frame.
        :param frame: numpy array representing an image
        :param wait: time to wait for a key press (ms)
        :return: True if there is no keyboard callback and user pressed 'q', False otherwise
        """
        if not self._started:
            self._init_display((frame.shape[1], frame.shape[0]))

        t_start = time.perf_counter()

        self._fps_info['t_idle'] += t_start - self._t_display_complete
        cv2.imshow(self._window_name, frame)
        k = cv2.waitKey(1) & 0xFF
        if self._keyboard_callback is not None:
            self._keyboard_callback(k)
        elif k == ord('q'):
            return True
        self._fps_info['n_frames'] += 1
        if t_start - self._fps_info['t_start'] > self._fps_info['reporting_cycle_seconds']:
            now = time.perf_counter()
            dt = now - self._fps_info['t_start']
            self._fps_info['fps'] = self._fps_info['n_frames']/dt
            self._fps_info['mean_idle_time'] = self._fps_info['t_idle'] / \
                self._fps_info['n_frames']
            self._fps_info['mean_busy_time'] = self._fps_info['t_busy'] / \
                self._fps_info['n_frames']
            if not self._quiet:
                t_total = self._fps_info['mean_idle_time'] + \
                    self._fps_info['mean_busy_time']
                logging.info("Display FPS: %f, mean display time/frame: %.3f ms (%.1f %%), mean idle/frame: %.3f ms (%.1f %%) " %
                             (self._fps_info['fps'],
                              1000*self._fps_info['mean_busy_time'],
                              100*(self._fps_info['mean_busy_time']/t_total),
                              1000*self._fps_info['mean_idle_time'],
                              100*(self._fps_info['mean_idle_time']/t_total)))
            self._fps_info['t_busy'] = 0.
            self._fps_info['t_idle'] = 0.
            self._fps_info['n_frames'] = 0
            self._fps_info['t_start'] = now

        self._t_display_complete = time.perf_counter()
        self._fps_info['t_busy'] += self._t_display_complete - t_start

        return False
