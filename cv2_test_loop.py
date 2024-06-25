"""
Create a 640x480 test pattern with fine detail and display it in a window.
"""
import cv2
import numpy as np
import logging
import time


def make_test_frame(w=640, h=480):
    """
    Make a frame with a chekerboard pattern, white and self._bkg_color, and each square has a side length of 100 pixels. 
    And draw 1-pixel by 1-pixel grid pattern in the upper left corner.
    """

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (255, 255, 255)
    for x in range(0, w, 100):
        for y in range(0, h, 100):
            if (x // 100) % 2 == (y // 100) % 2:
                frame[y:y+100, x:x+100] = (0, 0, 0)
    for x in range(0, w, 10):
        frame[0:10, x:x+10] = (0, 0, 0)
    for y in range(0, h, 10):
        frame[y:y+10, 0:10] = (0, 0, 0)

    # draw a grid with 1 pixel between each line in the first 100x100 square of the frame
    for x in range(0, 100, 1):
        frame[x, 0:100] = (0, 0, 0)
        frame[0:100, x] = (0, 0, 0)

    return frame


def test_loop():
    """
    Create a 640x480 test pattern with fine detail and display it in a window.
    """
    w = 1640
    h = 1080
    frame = make_test_frame(w, h)
    window_name = "Test Pattern"
    #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    n_frames = 0

    make_frame_t = 0
    display_frame_t = 0

    def make_frame():
        return frame.copy()

    t_start_loop = time.perf_counter()
    t_start = t_start_loop

    while True:
        new_frame = make_frame()
        now = time.perf_counter()
        make_frame_t += now - t_start_loop
        cv2.imshow(window_name, new_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        n_frames += 1
        t_start_loop = time.perf_counter()
        display_frame_t += t_start_loop - now

        if n_frames % 100 == 0:
            now = time.perf_counter()
            dt = now - t_start
            fps = n_frames / (dt)
            mean_disp_time = display_frame_t / n_frames * 1000
            mean_frame_time = make_frame_t / n_frames * 1000
            pct_time_displaying = display_frame_t / \
                (display_frame_t + make_frame_t) * 100
            pct_time_frame_making = make_frame_t / \
                (display_frame_t + make_frame_t)*100

            logging.info(f"FPS: {fps:.1f}, mean display time: {mean_disp_time:.3f} ms ({
                         pct_time_displaying:.3f} %), mean frame time: {mean_frame_time:.3f} ms,({pct_time_frame_making:.3f} %)")
            n_frames = 0
            make_frame_t = 0
            display_frame_t = 0
            t_start = now

        t_start_loop = time.perf_counter()  # don't include time for printing stats

    cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_loop()
