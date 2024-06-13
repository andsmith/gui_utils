"""
Represent possible camera settings & user's choice.
(See README.md for details)
"""

import cv2
import json
import logging
from .gui_picker import ChooseItemDialog, choose_item_text
from .detect_camera_settings import SystemCameraConfig
import os

# File created when user selects which of the settings detected are to be used  (delete it to ask user again, etc.)
_USER_CAMERA_SETTINGS = ".camera.user_config.json"


class UserSettingsManager(object):
    """
    Class to ask user for camera settings, remember for next time.
    """
    USER_INTERACTION_MODES = ['gui', 'console', 'quiet']
    _SETTINGS = ['index', 'mirrored', 'res']

    def __init__(self, index=None, res_w_h=None, mirrored=True, interaction='gui'):
        """
        First, load SYSTEM settings (SystemCameraConfig object) to see what cameras & modes are available.
        Second, Look for USER's camera settings file to see which are selected,
            Ask the user for whatever info (index, resolution) is missing from that file (or all if the file is not
            there).
        Third, update/create the user's camera settings file if necessary.

        TODO: provide functions to change resolution after camera has started.

        :param index:  camera index, overrides settings file if not None, else will ask user or use default
        :param res_w_h:  tuple w/(width, height), same default behavior as index,
        :param mirrored:  If True, images will be horizontally flipped, same default behavior as index,
        :param interaction:  how to find missing information, must be one of UserSettingsManager.USER_INTERACTION_MODES
        """

        if interaction not in UserSettingsManager.USER_INTERACTION_MODES:
            raise Exception("Interaction mode must be one of these: %s" % (UserSettingsManager.USER_INTERACTION_MODES,))
        self._system = SystemCameraConfig()
        self._settings = {'mirrored': mirrored,
                          'index': index,
                          'res': res_w_h}
        self._interact = interaction

        old_settings = UserSettingsManager._load_settings_file()
        self._disambiguate_settings(old_settings)
        self._write_settings_file()

    def _write_settings_file(self):
        user_file_path = os.path.expanduser(os.path.join('~', _USER_CAMERA_SETTINGS))
        with open(user_file_path, 'w') as outfile:
            json.dump(self._settings, outfile)

        logging.info("Wrote new user camera config file:  %s" % (user_file_path,))

    def _disambiguate_settings(self, loaded):
        """
        If camera settings were not specified in the constructor, use values from loaded file,
        else ask user/use defaults depending on interaction mode.

        NOTE:  Default values are defined here.
        """
        use_gui = self._interact == 'gui'  # assume console otherwise

        # resolve index
        if self._settings['index'] is None:
            if loaded['index'] is None:
                if self._interact != 'quiet':
                    self._settings['index'] = user_pick_camera(use_gui)
                else:
                    self._settings['index'] = 0
            else:
                self._settings['index'] = loaded['index']

        # resolve if horizontally flipped
        if self._settings['mirrored'] is None:
            self._settings['mirrored'] = loaded['mirrored'] if loaded['mirrored'] is not None else True

        # resolve resolution
        valid_resolutions = self._system.get_cam_resolutions(self._settings['index'])
        if self._settings['res'] is None:
            if loaded['res'] is None:
                if self._interact != 'quiet':
                    self._settings['res'] = user_pick_resolution(valid_resolutions, use_gui)
                else:
                    self._settings['res'] = 640, 480
            else:
                self._settings['res'] = loaded['res']

    @staticmethod
    def _load_settings_file():
        user_file_path = os.path.expanduser(os.path.join('~', _USER_CAMERA_SETTINGS))
        if os.path.exists(user_file_path):
            logging.info("Found user camera settings file, loading...")
            with open(user_file_path, 'r') as infile:
                old_settings = json.load(infile)
        else:
            logging.info("User camera settings file not found.")
            old_settings = {setting: None for setting in UserSettingsManager._SETTINGS}
        return old_settings

    def is_mirrored(self):
        return self._settings['mirrored']

    def get_index(self):
        return self._settings['index']

    def get_resolution_wh(self):
        return self._settings['res']

    '''
    def change_settings(self, new_settings):
        """
        Enqueue camera settings changes (take effect before grabbing next frame)
        :param new_settings: dict(setting_name=setting_value, ...)
        """
        self._

    def flush_changes(self, cam):
        """
        Apply queued setting changes to camera.
        :param cam: VideoCapture object
        """
        things_to_set = {}
        while not self._settings_changes_q.empty():
            things_to_set.update(self._settings_changes_q.get(block=True))

        for setting in things_to_set:
            name = self._PROPS[setting]
            logging.info("Setting camera property '%s' (%i):  %s" % (name, setting, things_to_set[setting]))
            cam.set(setting, things_to_set[setting])

        for setting in things_to_set:
            new_value = cam.get(setting)
            name = self._PROPS[setting]
            logging.info("New camera property '%s' (%i):  %s" % (name, setting, new_value))

    '''


def user_pick_camera(n_cams, gui=True):
    """
    If there's more than one camera, ask user which camera to use.
    :param n_cams:  Number of available cameras (from e.g. SystemCameraConfig.get_n_cameras())
    :param gui:  If true, ask in dialog box, else use the console.
    :returns: integer, the camera index
    """
    prompt = "Please select one of the detected cameras:"
    if n_cams < 1:
        raise Exception("No cameras found on system.")
    elif n_cams == 1:
        cam_ind = 0
    else:
        choices = ['camera 0 (for laptops, probably user-facing)',
                   'camera 1 (probably forward-facing)']
        choices.extend(["camera %i" % (ind + 2,) for ind in range(n_cams - 2)])
        if gui:
            chooser = ChooseItemDialog(prompt=prompt)
            cam_ind = chooser.ask_text(choices)
        else:
            cam_ind = choose_item_text(choices, prompt)
        if cam_ind is None:
            raise Exception("User selected no camera.")
    return cam_ind


def user_pick_resolution(resolution_list, gui=True):
    """
    Prompt user to pick from a list of camera resolutions.
     gui=True, use TK dialog box, else use command prompt.
    :param resolution_list: dict(widths=[w0, w1, ...], heights=[h0, h1, ...])
    :param gui:  If true, ask in dialog box, else use the console.
    :return:  (width, height) or None if user opted out of selection.
    """
    choices = ["%i x %i" % (w, h) for w, h in zip(resolution_list['widths'], resolution_list['heights'])]
    if gui:
        selection = ChooseItemDialog(prompt="Choose one of the detected\ncamera resolutions:").ask_text(choices=choices)
    else:
        selection = choose_item_text(prompt="Choose one of the detected\ncamera resolutions:", choices=choices)

    return resolution_list['widths'][selection], resolution_list['heights'][selection]


def _user_test():
    usm = UserSettingsManager()
    logging.info("UserSettingsManager created with:\n\tcamera:  %i\n\tmirrored:  %s\n\tresolution:  %s" % (
        usm.get_index(),
        usm.is_mirrored(),
        usm.get_resolution_wh()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # _interactive_test()

    _user_test()
