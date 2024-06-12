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
from .platform_deps import open_camera, is_v_10, is_windows

_APP_NAME = "gui_utils"
_APP_AUTHOR = "andsmith"

# in user dir (e.g. .local)
_SYSTEM_CAMERA_SPECS = ".camera.system_config.json"  # Created when computer & camera are scanned

# in app src dir
_RES_FILE_SHORT = "common_resolutions_abbrev.json"  # scan for these if Win 10, only the more common
_RES_FILE = "common_resolutions.json"  # scan for these otherwise


class SystemCameraConfig(object):
    """
    Represent system's camera capabilities: how many cameras, what resolutions they are capable of.
    Scan just in time, wherever possible.
    """

    def __init__(self, scan=True):
        """
        Attempt to load system config file, otherwise use defaults and scan the computer.
        :param scan:  If true, scan system for camera capabilities as that info is required.
                      if false, assume any camera referenced exists and has default resolution capabilities.
                      Write results to ~/.camera.system_config.json,
                      (Ignored if ~/.camera.system_config.json exists and contains this info)
        """
        self._scan = scan

        # Store system camera info in a dict here, never store defaults as if they were scanned values.
        self._cameras = None
        #   if self._cameras = None  -->   computer has not been scanned for cameras.
        #   if self._cameras[i] = None  -->  camera_i has not been scanned for resolution capabilities.
        #   else self._cameras[i] = dict( widths: [..], heights[..]) possible resolutions for cam i.

        self._default_resolutions = SystemCameraConfig._load_default_resolutions()

        # load system info if it exists
        config_file_path = os.path.expanduser(os.path.join('~', _SYSTEM_CAMERA_SPECS))
        if os.path.exists(config_file_path):
            logging.info("Found system camera configuration file, loading...")
            self._cameras = _read_sys_config(config_file_path)
        else:
            logging.info("System camera configuration file not found...")

            if scan:
                logging.info("\t... scanning for system camera configuration")
                n = self._count_system_cameras()
                logging.info("\t... found %i cameras." % (n,))
            else:
                logging.info("Scan disabled, using default camera definitions.")
                self._cameras = None

            logging.info("Creating new local system configuration file...")
            self.write_config()

        camera_scan_status = "unscanned" if self._cameras is None else "%i cameras" % (len(self._cameras),)
        logging.info("Current system camera configuration:  %s" % (camera_scan_status,))
        # self._cameras exists now, but may be None or have None-entries for some/all camera indices.

    def _count_system_cameras(self):
        """
        See how many cameras are attached to this computer.
        Initialize data structures if this is the first scan.
        """
        n_cams = count_cameras()
        if self._cameras is None:  # this was the first scan,
            self._cameras = {i: None for i in range(n_cams)}
        return n_cams

    def get_n_cameras(self):
        """
        Return the number of cameras the OS can see, or None if this computer has not been scanned and scanning
        is disabled.
        """
        if self._cameras is None:
            if not self._scan:
                return None
            else:
                self._count_system_cameras()

        return len(self._cameras)

    def write_config(self):
        config_file_path = os.path.expanduser(os.path.join('~', _SYSTEM_CAMERA_SPECS))
        with open(config_file_path, 'w') as outfile:
            json.dump(self._cameras, outfile)
        logging.info("Wrote config file:  %s" % (config_file_path,))

    def get_cam_resolutions(self, cam_index):
        """
        What resolutions is the camera[cam_index] capable of?
        If the info is missing, scan for it or return defaults
        """
        if self._cameras is None:
            if not self._scan:
                return self._default_resolutions
            else:
                self._cameras = {cam_index: probe_resolutions(self._default_resolutions, cam_index)}
                self.write_config()
                return self._cameras[cam_index]
        else:
            if cam_index not in self._cameras or self._cameras[cam_index] is None:
                if not self._scan:
                    return self._default_resolutions
                else:
                    self._cameras = {cam_index: probe_resolutions(self._default_resolutions, cam_index)}
                    self.write_config()
            return self._cameras[cam_index]

    @staticmethod
    def _load_default_resolutions():
        if is_windows() and is_v_10():
            res_data_file = _RES_FILE_SHORT
        else:
            res_data_file = _RES_FILE
        logging.info("Reading default camera resolutions file:  %s" % (res_data_file,))

        path = os.path.join(os.path.split(__file__)[0], res_data_file)
        with open(path, 'r') as infile:
            data = json.load(infile)
        return data


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
    return result


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


def _read_sys_config(filename):
    """
    Json will convert integer keys into strings, so they need to be converted back when loaded.
    (keys are the camera indices)
    """
    with open(filename, 'r') as infile:
        camera_info = json.load(infile)
    return {int(ind): camera_info[ind] for ind in camera_info}


def _sys_test():
    cams = SystemCameraConfig()
    logging.info("SystemSettings object created with %i cameras." % (cams.get_n_cameras(),))
    resolutions = cams.get_cam_resolutions(0)
    logging.info("Camera 0 has %i resolutions." % (len(resolutions['widths']),))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # _interactive_test()

    _sys_test()
