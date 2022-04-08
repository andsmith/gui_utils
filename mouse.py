"""
Hold state of mouse, for interacting with cv2 windows.
"""
import cv2
import keyboard
import enum


class MouseButtons(enum.Enum):
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2


class ModKeys(enum.Enum):
    SHIFT = 10
    ALT = 11
    CTRL = 12


class ButtonStates(enum.Enum):
    UP = False
    DOWN = True


class MouseState(object):
    MOUSE_BUTTON_EVENTS = {
        cv2.EVENT_LBUTTONDOWN: {'button': MouseButtons.LEFT, 'state': ButtonStates.DOWN, 'desc': "l-down"},
        cv2.EVENT_LBUTTONUP: {'button': MouseButtons.LEFT, 'state': ButtonStates.UP, 'desc': "l-up"},
        cv2.EVENT_RBUTTONDOWN: {'button': MouseButtons.RIGHT, 'state': ButtonStates.DOWN, 'desc': "r-down"},
        cv2.EVENT_RBUTTONUP: {'button': MouseButtons.RIGHT, 'state': ButtonStates.UP, 'desc': "r-up"},
        cv2.EVENT_MBUTTONDOWN: {'button': MouseButtons.MIDDLE, 'state': ButtonStates.DOWN, 'desc': "m-down"},
        cv2.EVENT_MBUTTONUP: {'button': MouseButtons.MIDDLE, 'state': ButtonStates.UP, 'desc': "m-up"}}

    KEYS = {ModKeys.SHIFT: 'shift',
            ModKeys.ALT: 'alt',
            ModKeys.CTRL: 'ctrl'}

    def __init__(self):
        self._position = None
        self._prev_position = None
        self._mouse_button_states = {m: ButtonStates.UP for m in MouseButtons}  # maps button enum to click location
        self._key_states = {}  # maps key enums to True (down) or False (up)
        self._update_keyboard_state()

    def _update_keyboard_state(self):
        self._key_states = {k: keyboard.is_pressed(self.KEYS[k]) for k in self.KEYS}

    def get_state(self):
        self._update_keyboard_state()
        return {'mouse_buttons': self._mouse_button_states,
                'mod_keys': self._key_states,
                'mouse_position': self._position}

    def update_state(self, event, x, y, flags, param):
        """
        Call this inside your CV2 mouse event callback, with the same params
        """
        self._prev_position = self._position
        self._position = x, y
        self._update_keyboard_state()
        motion, position, button_change = None, None, None

        if self._prev_position is not None:
            motion = self._position[0] - self._prev_position[0], \
                     self._position[1] - self._prev_position[1]

        if event in self.MOUSE_BUTTON_EVENTS:  # up or down click
            button_info = self.MOUSE_BUTTON_EVENTS[event]
            self._mouse_button_states[button_info['button']] = button_info['state']
            button_change = button_info['desc']

        return {'mouse_buttons': self._mouse_button_states,
                'mod_keys': self._key_states,
                'mouse_position': self._position,
                'button_change': button_change,
                'motion': motion}
