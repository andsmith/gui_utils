"""
Hold state of mouse, for interacting with cv2 windows.
"""
import cv2


class KeyboardState(object):
    def __init__(self):
        self.shift = 0
        self.ctrl = 0
        self.alt = 0

    def update_state(self, k):
        pass


class MouseState(object):
    _MOUSE_BUTTON_EVENTS = {cv2.EVENT_LBUTTONDOWN: {'ind': 0, 'state': 1, 'desc': "l-down"},
                            cv2.EVENT_LBUTTONUP: {'ind': 0, 'state': 0, 'desc': "l-up"},
                            cv2.EVENT_RBUTTONDOWN: {'ind': 1, 'state': 1, 'desc': "r-down"},
                            cv2.EVENT_RBUTTONUP: {'ind': 1, 'state': 0, 'desc': "r-up"},
                            cv2.EVENT_MBUTTONDOWN: {'ind': 2, 'state': 1, 'desc': "m-down"},
                            cv2.EVENT_MBUTTONUP: {'ind': 2, 'state': 0, 'desc': "m-up"}}

    def __init__(self):
        self._x, self._y = None, None
        self._last_x, self._last_y = None, None
        self._buttons = [0, 0, 0]

    def get_button_state(self):
        return self._buttons

    def update_state(self, event, x, y, flags, param):
        """
        Call this inside your CV2 mouse event callback, with the same params
        """
        self._last_x, self._last_y = self._x, self._y
        self._x, self._y = x, y
        motion, position, button_change = None, None, None
        if self._x is not None:
            position = self._x, self._y
            if self._last_x is not None:
                motion = self._x - self._last_x, self._y - self._last_y
        if event in self._MOUSE_BUTTON_EVENTS:
            self._buttons[self._MOUSE_BUTTON_EVENTS[event]['ind']] = \
                self._MOUSE_BUTTON_EVENTS[event]['state']
            button_change = self._MOUSE_BUTTON_EVENTS[event]['desc']
        return position, motion, button_change, self._buttons
