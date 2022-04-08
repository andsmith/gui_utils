from .gui_picker import make_test_image
from .text_annotation import StatusMessages
import cv2


def _test_status_msgs():  # interactive test
    base_img = make_test_image((480, 640, 3), n_rects=100)

    m = StatusMessages(img_shape=base_img.shape,
                                       text_color=[255, 255, 255],
                                       bkg_color=[255, 40, 0],
                                       bkg_alpha=.65441, spacing=5)
    m.add_msg("This message should disappear after 5 seconds.", "msg 1", duration_sec=5.0)
    m.add_msg("This message should disappear after 7 seconds and it's really long, so it should be smaller.",
              "msg 2", duration_sec=7.0)
    m.add_msg("This message should stay forever (q to quit).", "msg 3", duration_sec=0)
    m.add_msg("This message should disappear after 10 seconds.", "msg 4", duration_sec=10.0)
    while True:

        frame = base_img.copy()
        m.annotate_img(frame)
        cv2.imshow("output", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _test_status_msgs()
