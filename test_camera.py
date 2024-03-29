from .camera import Camera, pick_camera
import logging
import time
import cv2


class CamTester(object):
    """Open a camera and show the video stream."""

    def __init__(self, cam_index=0, settings=None):
        self._n_frames = 0
        self._print_interval = 30
        self._t_start = None
        self._cam_index = cam_index
        self._cam = Camera(self._cam_index, self._show_img, settings=settings, prompt_resolution=True)
        self._cam.start()

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
    # cam_ind = pick_camera(gui=False)
    cam_ind = pick_camera(gui=True)
    print("Done testing camera picker.")
    CamTester(cam_ind)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_camera()
