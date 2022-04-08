import camera
import logging


def _test_camera():
    #cam_ind = camera.pick_camera(gui=False)
    cam_ind = camera.pick_camera(gui=True)
    print("Done testing camera picker.")
    camera.CamTester(cam_ind)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_camera()
