"""
Interactive coordinate grids, i.e. cartesian, log/log, polar etc.
For continuous 2-d user input control, etc.
"""
import logging
import cv2
import time
import numpy as np
from abc import abstractmethod, ABCMeta

RGB_COLORS = {'white': (255, 255, 255),
              'light_gray': (184, 184, 184),
              'gray': (128, 128, 128),
              'dark_gray': (20, 20, 20)}


def _rgb_to_bgra(rgb, a):
    return rgb + (a,)


class Grid(metaclass=ABCMeta):
    DEFAULT_DRAW_PROPS = {'line_thicknesses': {'thick': 3,
                                               'medium': 2,
                                               'thin': 1},
                          'font': cv2.FONT_HERSHEY_SIMPLEX,
                          'font_scale': 0.4,
                          'tick_length': 10,
                          'draw_bounding_box': True,
                          'tick_string': "%.2g",
                          'crosshair_length': 20,
                          'n_ticks': 9}

    DEFAULT_COLORS = {'bkg': _rgb_to_bgra(RGB_COLORS['dark_gray'], 255),
                      'heavy': _rgb_to_bgra(RGB_COLORS['white'], 255),
                      'light': _rgb_to_bgra(RGB_COLORS['light_gray'], 255),
                      'medium': _rgb_to_bgra(RGB_COLORS['gray'], 255), }

    HOTKEYS = [{'range_up': ']',  # param 1 adjust
                'range_down': '['},
               {'range_up': '0',  # param 0 adjust
                'range_down': 'p'}]

    def __init__(self, bbox, param_ranges=((0., 1.), (0., 1.)), colors=None, draw_props=None, expansion_speed=1.33):
        """
        Initialize a new grid.
        :param bbox: dict with 'top','bottom','left','right', where in image is the grid region
        :param param_ranges:  initial range of grid params
        :param colors:  Dict with RGB or RGBA values, {'heavy','medium','light','bkg'}
        :param draw_props:  dict with info for drawing (see DEFAULT_DRAW_PROPS)
        :param expansion_speed:  How fast to grow/shrink when user wants to reshape grid
        """

        # use defaults where not specified
        self._props = Grid.DEFAULT_DRAW_PROPS.copy()
        self._props.update({} if draw_props is None else draw_props)

        self._param_ranges = np.array(param_ranges)

        self._param_spans = self._param_ranges[:, 1] - self._param_ranges[:, 0]

        self._calc_ticks()

        self._param_values = None, None
        self._bbox = bbox
        self._x = expansion_speed
        self._mouse_pos = None
        self._image_offset = np.array([self._bbox['left'], self._bbox['top']])
        self._size = np.array([bbox['right'] - bbox['left'], bbox['bottom'] - bbox['top']])

        self._colors = Grid.DEFAULT_COLORS.copy()
        self._colors.update({} if colors is None else colors)

    def mouse(self, event, x, y, flags, param):
        """
        Mouse update function, CV2 style.
        Determine local unit coordinates.
        """
        self._mouse_pos = x, y

    def keyboard(self, k):

        # check for range adjustments
        for param_i, hotkey in enumerate(Grid.HOTKEYS):
            if k & 0xff == ord(hotkey['range_up']):
                self._param_ranges[param_i] *= self._x
                self._param_spans = self._param_ranges[:, 1] - self._param_ranges[:, 0]
                self._calc_ticks()
            elif k & 0xff == ord(hotkey['range_down']):
                self._param_ranges[param_i] /= self._x
                self._param_spans = self._param_ranges[:, 1] - self._param_ranges[:, 0]
                self._calc_ticks()

    @abstractmethod
    def _calc_ticks(self):
        """
        set tick mark locations, text, text sizes
        """
        pass

    def get_values(self):
        return self.pixel_to_grid_coords(self._mouse_pos)

    @abstractmethod
    def pixel_to_grid_coords(self, xy_pos):
        """
        Given the grid image coordinates, what is the grid position.
        """
        pass

    @abstractmethod
    def draw(self, image):
        """
        draw grid within self._bbox of image
        """
        pass

    def draw_crosshair(self, image):
        if self._mouse_pos is None:
            return
        x_px, y_px = self._mouse_pos
        l = int(self._props['crosshair_length'] / 2)
        _draw_rect(image, x_px, y_px - l, 1, l * 2, self._colors['heavy'])
        _draw_rect(image, x_px - l, y_px, l * 2, 1, self._colors['heavy'])

    def _draw_base_image(self, image):
        """
        helper to draw common elements
        """
        # background
        _draw_rect(image, self._bbox['left'], self._bbox['top'],
                   self._size[0], self._size[1],
                   self._colors['bkg'])
        # box
        if self._props['draw_bounding_box']:
            t = self._props['line_thicknesses']['medium']
            _draw_rect(image, self._bbox['left'], self._bbox['top'], self._size[0], t, self._colors['heavy'])
            _draw_rect(image, self._bbox['left'], self._bbox['bottom'] - t, self._size[0], t, self._colors['heavy'])
            _draw_rect(image, self._bbox['right'] - t, self._bbox['top'], t, self._size[1], self._colors['heavy'])
            _draw_rect(image, self._bbox['left'], self._bbox['top'], t, self._size[1], self._colors['heavy'])


class CartesianGrid(Grid):
    def __init__(self, bbox, **kwargs):
        super(CartesianGrid, self).__init__(bbox, **kwargs)

    def pixel_to_grid_coords(self, xy_pos):
        pos_rel = (np.array(xy_pos) - self._image_offset) / self._size
        pos_rel[1] = 1.0 - pos_rel[1]  # flip y
        param_range_lengths = self._param_ranges[:, 1] - self._param_ranges[:, 0]
        return pos_rel * param_range_lengths + self._param_ranges[:, 0]

    def draw(self, image):
        """
        Draw grid.
        """
        self._draw_base_image(image)

        tick_thickness = self._props['line_thicknesses']['thin']
        tick_color = self._colors['heavy']
        tick_length = self._props['tick_length']

        # vertical ticks
        x_left = self._bbox['left']
        x_right = self._bbox['right'] - tick_length
        x_label = int(x_left + tick_length * 1.33)

        for y_i, y_grid in enumerate(self._param_ticks[1]):
            y_rel = 1 - (y_grid - self._param_ranges[1][0]) / self._param_spans[1]
            y_px = int((y_rel * self._size[1]) + self._bbox['top'])
            _draw_rect(image, x_left, y_px, tick_length, tick_thickness, tick_color)
            _draw_rect(image, x_right, y_px, tick_length, tick_thickness, tick_color)
            # labels
            y_label = int(y_px + self._tick_text_sizes[1][y_i][0][1] / 2)
            cv2.putText(image, "%g" % y_grid, (x_label, y_label), self._props['font'],
                        self._props['font_scale'], tick_color, 1, cv2.LINE_AA)

        # horizontal ticks
        y_top = self._bbox['top']
        y_bottom = self._bbox['bottom'] - tick_length
        y_label = int(y_bottom - tick_length * 1.33)

        for x_i, x_grid in enumerate(self._param_ticks[0]):
            x_rel = (x_grid - self._param_ranges[0][0]) / self._param_spans[0]
            x_px = int((x_rel * self._size[0]) + self._bbox['left'])
            _draw_rect(image, x_px, y_top, tick_thickness, tick_length, tick_color)
            _draw_rect(image, x_px, y_bottom, tick_thickness, tick_length, tick_color)
            # labels
            x_label = int(x_px - self._tick_text_sizes[0][x_i][0][0] / 2)

            cv2.putText(image, "%g" % x_grid, (x_label, y_label), self._props['font'],
                        self._props['font_scale'], tick_color, 1, cv2.LINE_AA)

    def _calc_ticks(self):
        self._param_ticks = [np.linspace(r[0], r[1], self._props['n_ticks'] + 2)[1:-1] for r in self._param_ranges]
        self._tick_text_sizes = [[cv2.getTextSize(self._props['tick_string'] % (tick_value,),
                                                  self._props['font'],
                                                  self._props['font_scale'],
                                                  thickness=1)
                                  for tick_value in ticks] for ticks in self._param_ticks]


def _draw_rect(image, left, top, width, height, color):
    image[top:top + height, left:left + width] = color


def grid_sandbox():
    blank = np.zeros((700, 1000, 4), np.uint8)
    blank[:, :, 3] = 255
    bbox = {'top': 10, 'bottom': 690, 'left': 10, 'right': 990}
    grid = CartesianGrid(bbox)
    window_name = "Grid sandbox"

    def _mouse(event, x, y, flags, param):
        grid.mouse(event, x, y, flags, param)

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse)
    draw_times = []
    frame_count = 0
    while True:
        frame = blank.copy()

        t_start = time.perf_counter()
        grid.draw(frame)
        grid.draw_crosshair(frame)
        dt = time.perf_counter() - t_start

        cv2.imshow(window_name, frame)
        k = cv2.waitKey(1)
        grid.keyboard(k)
        if k & 0xff == ord('q'):
            break

        draw_times.append(dt)
        frame_count += 1

        if frame_count % 100 == 0:
            print("Mean draw time:  %.6f sec (sd. %.6f sec)." % (np.mean(draw_times), np.std(draw_times),))
            frame_count = 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    grid_sandbox()
