"""
Generate frames of a solid color, send them somewhere periodically.
(i.e. as a substitute for a Camera() object when not using a camera.)
"""

import numpy as np
import logging
from .colors import RGB_COLORS as COLORS
from .video_io_base import VideoBase
from .gui_picker import ChooseItemDialog, choose_item_text
import time
import cv2


class BlankFrameGenerator(VideoBase):
    """
    Generate frames of a solid color, for when not using a camera.
    """
    COMMON_RESOLUTIONS = [(640, 480), (800, 600),
                          (1024, 768), (1280, 720), (1920, 1080)]

    def __init__(self, frame_res=None, disp_res=None, window_name="Video Base",
                 callback=None, mouse_callback=None, keyboard_callback=None,
                 window_flags=cv2.WINDOW_AUTOSIZE,  dt=1.0/30.0, color=COLORS['dark_dark_gray'], ask_user='gui'):
        """
        :param frame_res:  Tuple with (width, height), resolution of frames to generate, or None (see param ask_user).
        :param disp_res:  Tuple with (width, height), resolution of window to display frames in, or None to use res_in / default.

        :param callback:  function(frame, time) to call with new frames
        :param mouse_callback:  function(event, x, y, flags, param) to call with mouse events
        :param keyboard_callback:  function(key) to call with keyboard events
        :param window_name:  Name of the window to display the camera feed in.
        :param dt:  Time between frames, in seconds.
        :param color:  Color of the background, as an RGB tuple.
        :param ask_user:  'gui' to prompt user for settings in a dialog box, 'cli' to ask on the command line, 'silent'/None to use defaults.
        """
        self._ask_user = ask_user
        super().__init__(frame_res, disp_res, window_name,
                         callback=callback, mouse_callback=mouse_callback,
                         keyboard_callback=keyboard_callback, window_flags=window_flags)

        self._dt = dt
        self._bkg_color = color
        self._make_frame()
        # for timing the delay between frames
        self._last_frame_timestamp = time.perf_counter()

    def _set_camera_resolution(self, res):
        pass

    def _make_frame(self):
        """
        Make a frame with a chekerboard pattern, white and self._bkg_color, and each square has a side length of 100 pixels. 
        And draw 1-pixel by 1-pixel grid pattern in the upper left corner.
        """
        w, h = self._frame_res
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)
        self._frame[:, :] = self._bkg_color

    def _make_test_frame(self):
        """
        Make a frame with a chekerboard pattern, white and self._bkg_color, and each square has a side length of 100 pixels. 
        And draw 1-pixel by 1-pixel grid pattern in the upper left corner.
        """
        side = 100
        w, h = self._frame_res
        self._frame = np.zeros((h, w, 3), dtype=np.uint8)
        self._frame[:, :] = (255, 255, 255)
        # for x_ind, i in enumerate(range(0, h, side)):
        # for y_ind, j in enumerate(range(0, w, side)):
        #       if (x_ind + y_ind) % 2 == 0:
        #            self._frame[i:i+side, j:j+side] = self._bkg_color

        grid_color = (32, 255, 20)
        black = (0, 0, 0)

        def draw_grid(square_offset_ij, color, thickness):
            """
            Draw a grid pattern in the specified corner of the checkerboard.
            :param square_offset_ij:  2-tuple of square offsets from the upper left corner of the frame, in multiples of side.
            :param color:  RGB tuple
            :param thickness:  int, thickness of the lines in pixels
            """
            x_offset, y_offset = square_offset_ij[0] * \
                side, square_offset_ij[1]*side
            for i in range(0, side, thickness*2):
                self._frame[i+y_offset, x_offset:x_offset+side] = color
                self._frame[y_offset:y_offset+side, i+x_offset] = color

        draw_grid((0, 0),  black, 1)
        draw_grid((0, 1),  black, 2)
        draw_grid((1, 0),  black, 3)
        draw_grid((1, 1),  black, 4)
        draw_grid((0, 2),  grid_color, 8)
        draw_grid((2, 0),  grid_color, 16)
        draw_grid((2, 2),  grid_color, 32)

    def _disambiguate_resolution(self, frame_res, disp_res):
        """
        If frame_res is None...
        :param ask_user:  'gui' to prompt user for settings in a dialog box, 'cli' to ask on the command line, 'silent'/None to use defaults.
        :param frame_res: constructor's arg
        :param disp_res: constructor's arg
        """
        if frame_res is None:
            if self._ask_user == 'gui':
                resp = ChooseItemDialog(prompt="Choose resolution").ask_text(
                    choices=BlankFrameGenerator.COMMON_RESOLUTIONS)
                frame_res = BlankFrameGenerator.COMMON_RESOLUTIONS[resp]
            elif self._ask_user == 'cli':
                resp = choose_item_text(
                    prompt="Choose resolution: ", choices=BlankFrameGenerator.COMMON_RESOLUTIONS)
                frame_res = BlankFrameGenerator.COMMON_RESOLUTIONS[resp]
            else:
                frame_res = BlankFrameGenerator.COMMON_RESOLUTIONS[0]
        disp_res = frame_res if disp_res is None else disp_res

        logging.info("Using resolutions:  %s (generated frames), and %s (display window)" % (
            frame_res, disp_res))

        return frame_res, disp_res

    def _stop_making_frames(self):
        # nothing to stop, self._shdutdown will be set to True by base class
        pass

    def _start_making_frames(self):
        logging.info("Starting background frame generator")
        while not self._stop:
            # time.sleep(self._dt)
            if self._callback:
                frame = self._frame.copy()
                self._callback(frame, time.perf_counter())


def camera_faker_tester():
    """
    Open a frame generator that asks the user for a resolution, then opens a Display to show the solid color frames.
    """
    video = BlankFrameGenerator(
        window_name="Fake Camera tester - Press 'q' to quit", ask_user='cli')

    def show_frame(frame, time):
        if video.show(frame):
            video.shutdown()

    video.set_frame_callback(show_frame)
    video.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    camera_faker_tester()
