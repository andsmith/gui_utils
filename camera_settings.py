"""
Utilities / GUI for changing these webcam settings:
    * resolution
    * ?
"""

import pandas as pd
import cv2
import ssl
import os
import json
import logging
from .gui_picker import ChooseItemDialog, choose_item_text
from .platform_deps import open_camera

_RESOLUTION_CACHE_FILENAME = "common_resolutions.json"


def _get_cache_filename():
    path, _ = os.path.split(__file__)
    return os.path.join(path, _RESOLUTION_CACHE_FILENAME)


def download_common_resolutions(save_file=None):
    """
    Get list from wikipedia.
    idea from: https://www.learnpythonwithrune.org/find-all-possible-webcam-resolutions-with-opencv-in-python/

    :param save_file:  save list (JSON), i.e. to make a new local cached copy
    :return: dict(widths=[list of widths], heights = [list of heights])
    """
    logging.info("Downloading resolution list...")

    # https://stackoverflow.com/questions/44629631/while-using-pandas-got-error-urlopen-error-ssl-certificate-verify-failed-cert
    ssl._create_default_https_context = ssl._create_unverified_context
    url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"

    table = pd.read_html(url)[0]
    table.columns = table.columns.droplevel()
    resolutions = table.get(['W', 'H'])
    widths, heights = resolutions['W'].tolist(), resolutions['H'].tolist()
    resolutions = {'widths': widths, 'heights': heights}
    if save_file is not None:
        with open(save_file, 'w') as outfile:
            json.dump(resolutions, outfile)
        logging.info("Wrote %i resolutions to file:  %s" % (len(widths), save_file))
    return resolutions


def get_common_resolutions(no_network=False):
    """
    Try to download and/or load local copy.

    :param no_network:  skip download attempt
    :return: dict(widths=[list of widths], heights = [list of heights])
    """
    load_cache = False
    if not no_network:
        logging.info("Attempting to download list of common resolutions...")
        try:
            resolutions = download_common_resolutions()
            logging.info("Found %i resolutions." % (len(resolutions['widths']),))
        except:
            logging.info("Download failed, loading cached resolutions...")
            load_cache = True
    else:
        load_cache = True
    if load_cache:
        with open(_get_cache_filename(), 'r') as infile:
            resolutions = json.load(infile)
        logging.info("Loaded %i resolutions from cache." % (len(resolutions['widths']),))
    return resolutions


def probe_resolutions(resolutions, cam_index):
    """
    See which resolutions camera can support
    :param resolutions: dict(widths=[list of widths], heights = [list of heights])
    :param cam_index: which camera?
    :return: dict(widths=[list of widths], heights = [list of heights])
    """

    def _test(w, h):
        print("\tprobing %i x %i ..." % (w, h))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        worked = (width == w and height == h)
        print("\t%s" % (worked,))
        return worked

    logging.info("Probing camera %i ..." % (cam_index,))
    came = open_camera(cam_index)
    valid = [(w, h) for w, h in zip(resolutions['widths'], resolutions['heights']) if _test(w, h)]
    cam.release()

    logging.info("Found %i valid resolutions." % (len(valid),))
    return {'widths': [v[0] for v in valid], 'heights': [v[1] for v in valid]}


def user_pick_resolution(camera_index=0, no_network=False, gui=True):
    """
    Download (or load if offline) list of common resolutions.
    Check camera can be set to each one.
    Prompt user to pick one.

    if gui=True, use TK dailog box, else use command prompt.

    :param camera_index:  Which camera?
    :param no_network:  Skip download attempt
    :return:  (width, height) or None if user opted out of selection.
    """

    logging.info("\nLoading list of common resolutions...")
    res = get_common_resolutions(no_network=no_network)
    logging.info("\nProbing camera %i with %i resolutions...\n" % (camera_index, len(res['widths']),))
    valid = probe_resolutions(res, cam_index=camera_index)
    logging.info("\n\t... found %i valid resolutions!" % (len(valid['widths']),))

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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
