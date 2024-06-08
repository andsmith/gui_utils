"""
Represent possible camera settings & user's choice.
(See README.md for details)
"""

import pandas as pd
import cv2
import ssl
import os
import json
import logging
from .gui_picker import ChooseItemDialog, choose_item_text
from .platform_deps import open_camera, is_v_10, is_windows
import appdirs

_APP_NAME = "gui_utils"
_APP_AUTHOR = "andsmith"

# in user dir (e.g. .local)
_SYSTEM_CAMERA_SPECS = "camera_specs.json"  # Created when computer & camera are scanned
_USER_CAMERA_SETTINGS = "camera_config.json"  # Created when user selects which of the settings detected are to be used  (delete this to change, etc.)

# in app src dir
_RES_FILE_SHORT = "common_resolutions_abbrev.json"  # scan for these if Win 10, only the more common
_RES_FILE = "common_resolutions.json"  # scan for these otherwise


class SystemSettings(object):
    """
    Class to represent system camera info.
    Attempt to load system config file, otherwise use defaults and scan the computer.
    """

    def __init__(self):
        self._camera_resolutions = 
        self._default_resolutions = self._load_default_resolutions()

    def get_possible_resolutions(self, index):
        if index not in self._camera_resolutions:
            self._camera_resolutions[index] = probe_resolutions()

    def _load_default_resolutions(self):
        if is_windows() and is_v_10():
            res_data_file = _RES_FILE_SHORT
        else:
            res_data_file = _RES_FILE

        path = os.path.join(os.path.split(__file__)[0], res_data_file)
        with open(path, 'r') as infile:
            data = json.load(infile)
        return data


class CameraSettings(object):

    def __init__(self, index=None, res_w_h=None, mirrored=True, use_gui=True):
        """
        Manage
        """
        self._mirrored = mirrored
        self._gui = use_gui
        self._cam_index = index
        self._res = res_w_h

    def is_mirrored(self):
        return self._mirrored

    def _save_settings(self):
        pass

    def get_index(self):
        if self._cam_index is None:
            self._cam_index = user_pick_camera(self._gui)
        self._save_settings()
        return self._cam_index

    def get_resolution_wh(self):
        if self._res is None:
            self._res = user_pick_resolution(self.get_index(), self._gui)
        self._save_settings()
        return self._res

    def change_settings(self, new_settings):

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


def probe_resolutions(resolutions, cam_index):
    """
    See which resolutions camera can support
    :param resolutions: dict(widths=[list of widths], heights = [list of heights])
    :param cam_index: which camera?
    :return: dict(widths=[list of widths], heights = [list of heights])
    """

    def _test(cam, w, h):
        logging.info("\tProbing camera %i with %i x %i ..." % (cam_index, w, h))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        worked = (width == w and height == h)
        if worked:
            logging.info("\t\tSuccess.")
        return worked

    logging.info("Probing camera %i ..." % (cam_index,))
    cam = open_camera(cam_index)
    valid = [(w, h) for w, h in zip(resolutions['widths'],
                                    resolutions['heights']) if _test(cam, w, h)]
    cam.release()

    logging.info("Found %i valid resolutions." % (len(valid),))
    result = {'widths': [v[0] for v in valid], 'heights': [v[1] for v in valid]}


def user_pick_camera(gui=True):
    """
    Ask user which camera to use.
    """
    prompt = "Please select one of the detected cameras:"
    print("Detecting cameras...")
    n_cams = count_cameras()
    logging.info("Detected %i cameras." % (n_cams,))
    if n_cams < 1:
        raise Exception("Webcam required for this version")
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
        print("Chose", cam_ind)
    return cam_ind


def user_pick_resolution(camera_index=0, gui=True, probe=False):
    """
    Read list of common resolutions ("common_resolutions.json")
    Check camera can be set to each one.
    Prompt user to pick one.

    if gui=True, use TK dialog box, else use command prompt.

    :param camera_index:  Which camera?
    :return:  (width, height) or None if user opted out of selection.
    """

    logging.info("Loading list of resolutions...")
    res = read_resolutions_file()
    if probe:
        logging.info("\tProbing camera %i with %i resolutions...\n" % (camera_index, len(res['widths']),))
        valid = probe_resolutions(res, cam_index=camera_index)
        logging.info("\t\t... found %i valid resolutions!" % (len(valid['widths']),))
    else:
        valid = res
        logging.info("Not checking resolutions.")

    choices = ["%i x %i" % (w, h) for w, h in zip(valid['widths'], valid['heights'])]
    if gui:
        selection = ChooseItemDialog(prompt="Choose one of the detected\ncamera resolutions:").ask_text(choices=choices)
    else:
        selection = choose_item_text(prompt="Choose one of the detected\ncamera resolutions:", choices=choices)

    return valid['widths'][selection], valid['heights'][selection]


def count_cameras():
    """
    See how many cameras are attached to the computer.
    :return: number of cameras successfully opened

    WARNING:  Will not behave predictably if any cameras are in use.
    """
    n = 0
    c = None
    while True:
        try:
            c = open_camera(n)
            ret, frame = c.read()
            if frame.size < 10:
                raise Exception("Out of cameras!")
            c.release()
            cv2.destroyAllWindows()
            n += 1
        except:
            c.release()
            cv2.destroyAllWindows()
            break
    return n


def _interactive_test():
    # run to update cache:
    # update_common_resolutions(RESOLUTION_CACHE_FILENAME)
    # sys.exit()

    # run to demo picker
    # res = pick_resolution(1)
    # print("Selected:  ", res)
    # sys.exit()

    res = user_pick_resolution()
    print("User selected:  %s" % (res,))

    res = user_pick_resolution(gui=False)
    print("User selected:  %s" % (res,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _interactive_test()
