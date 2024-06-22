"""
Lightweight wrapper for CV2 display functions.
"""
import cv2
import time
import numpy as np
import logging
import threading


class Display(object):
    def __init__(self, window_name, size=(640, 480), window_flags=cv2.WINDOW_AUTOSIZE, quiet=False):
        self._window_name = window_name
        self._size = size
        self._frame = None
        self._quiet = quiet
        self._mouse_callback, self._keyboard_callback = None, None

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

        cv2.namedWindow(window_name, window_flags)

    def set_mouse_callback(self, callback=None):
        self._mouse_callback(callback)

    def set_keyboard_callback(self, callback=None):
        self._keyboard_callback(callback)

    def show(self, frame, wait=1):
        """
        Show a frame and handle keyboard input.
        :param frame: numpy array representing an image
        :param wait: time to wait for a key press (ms)
        :return: True if there is no keyboard callback and user pressed 'q', False otherwise
        """
        t_start = time.perf_counter()

        self._fps_info['t_idle'] += t_start - self._t_display_complete

        cv2.imshow(self._window_name, frame)
        k = cv2.waitKey(wait) & 0xFF
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
                logging.info("Display FPS: %f, mean idle time/frame: %f ms, mean display time/frame:  %f ms. total time: %f (of %f)" % (
                    self._fps_info['fps'],
                    1000*self._fps_info['mean_idle_time'],
                    1000*self._fps_info['mean_busy_time'],
                    self._fps_info['t_idle'] +
                    self._fps_info['t_busy'], dt))
                
            self._fps_info['t_busy'] = 0.
            self._fps_info['t_idle'] = 0.
            self._fps_info['n_frames'] = 0
            self._fps_info['t_start'] = now

        self._t_display_complete = time.perf_counter()
        self._fps_info['t_busy'] += self._t_display_complete - t_start

        return False

    def close(self):
        cv2.destroyWindow(self._window_name)


def test_display():
    w, h = (1600, 800)
    t0 = time.perf_counter()
    x_freq_range = [3, 8]
    y_freq_range = [2, 6]
    window_name = "Test Display"

    def _make_frame(size, time):
        x_freq = np.cos((time-t0)*2*np.pi) * \
            (x_freq_range[1]-x_freq_range[0])/2 + np.mean(x_freq_range)
        y_freq = np.sin(1+(time-t0)*2*np.pi) * \
            (y_freq_range[1]-y_freq_range[0])/2 + np.mean(y_freq_range)
        img = np.zeros((h, w), dtype=np.uint8)
        t = np.linspace(0, 2*np.pi, 10000)
        x = (np.cos(t*x_freq) + 1)/2 * (w-1)
        y = (np.sin(t*y_freq) + 1)/2 * (h-1)
        img[(y.astype(int), x.astype(int))] = 255
        return img

    display = Display(window_name)

    while True:
        frame = _make_frame((w, h), time.perf_counter())
        if display.show(frame):
            break
    display.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_display()
