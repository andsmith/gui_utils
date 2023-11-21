"""
Wrap pygame.camera, handle everything through callbacks.
UNFINISHED
"""


import pygame
import pygame.camera
import logging
import time
from loop_timing.loop_profiler import LoopPerfTimer


class VideoIO(object):

    def __init__(self, index=0, callbacks=None, res=(640, 480), fps=60):
        """
        General loop:
            1. Get frame,
            2. Send to callback, get frame to display,
            3. Display frame
        Asynchronously, send these to callbacks:
            * Mouse Events (as defined by pygame)
            * Keypress/release events

        :param index:  int, index into list returned by pygame.camera.list_cameras()
        :param callbacks:  Callback function for:
            'new_frame': function(img, frame, timestamp) returns img
            'keyboard': function(key, state, timestamp)
            'mouse': params(mouse_event, event_data, timestamp)
        :param res:  target resolution
        :param fps:  target frame rate, important if not using camera

        """
        self._shutdown = False
        self._callbacks = {} if callbacks is None else callbacks
        self._ind = index
        self._res = res
        self._fps_info = {'n': 0,
                          't_start': time.perf_counter(),
                          'interval': 100}
        self._fps = fps
        self._init_cam()
        self._init_disp()

    def _init_cam(self):
        pygame.init()
        pygame.camera.init()
        self._cam_list = pygame.camera.list_cameras()
        logging.info("Cameras found:\n\t%s" % ("\n\t".join(self._cam_list),))
        logging.info("Opening camera %i with resolution %s..." % (self._ind, self._res))
        self._cam = pygame.camera.Camera(self._cam_list[self._ind], self._res, 'rgb')
        self._size = self._cam.get_size()
        logging.info("... opened camera with resolution %s" % (self._size,))

    def _init_disp(self):
        self._disp = pygame.display.set_mode(self._size, 0)

    _CALLBACK_EVENTS = {pygame.KEYDOWN: 'keyboard',
                        pygame.KEYUP: 'keyboard',
                        pygame.MOUSEMOTION: 'mouse',
                        pygame.MOUSEBUTTONDOWN: 'mouse',
                        pygame.MOUSEBUTTONUP: 'mouse',
                        pygame.MOUSEWHEEL: 'mouse'}

    @LoopPerfTimer.time_function
    def _data_dispatch(self, event):
        if event.type in VideoIO._CALLBACK_EVENTS:
            kind = VideoIO._CALLBACK_EVENTS[event.type]
            if kind in self._callbacks:
                return self._callbacks[kind](event)
        return None

    @LoopPerfTimer.time_function
    def _handle_events(self):
        events = pygame.event.get()

        LoopPerfTimer.add_marker("Got Events")
        for e in events:
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and
                                         e.key == pygame.K_ESCAPE):
                self.stop()
            self._data_dispatch(e)
        LoopPerfTimer.add_marker("Processed Events")

    def start(self):
        LoopPerfTimer.reset(enable=False, burn_in=20, display_after=50)
        self._cam.start()

        frame = pygame.surface.Surface(self._size, 0, self._disp)

        n_ready = 0

        while not self._shutdown:
            LoopPerfTimer.mark_loop_start()
            self._handle_events()
            if self._cam.query_image():
                n_ready += 1
            frame = self._cam.get_image(frame)  # blocks
            LoopPerfTimer.add_marker("Got frame")
            # import ipdb; ipdb.set_trace()
            if 'frame' in self._callbacks:
                out_frame = self._callbacks['frame'](frame)

            else:
                out_frame = frame

            self._disp.blit(out_frame, (0, 0))
            LoopPerfTimer.add_marker("blit")

            pygame.display.flip()
            LoopPerfTimer.add_marker("flip")

            # monitor frame rate
            self._fps_info['n'] += 1
            if self._fps_info['n'] == self._fps_info['interval']:
                now = time.perf_counter()
                n, t_start = self._fps_info['n'], self._fps_info['t_start']
                fps = n / (now - t_start)
                logging.info("FPS:  %.3f  (%.3f ready)" % (fps, n_ready / n))
                self._fps_info['t_start'] = now
                self._fps_info['n'] = 0
                n_ready = 0

        logging.info("Main loop exiting")
        self._cam.stop()
        logging.info("Camera stopped.")

    def stop(self):
        self._shutdown = True


def cam_demo():
    v = VideoIO()
    v.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cam_demo()
