"""
Interactive coordinate grids, i.e. cartesian  (log, polar?)
For continuous 2-d user input control, etc.
"""
import logging
import cv2
import time
import numpy as np
from abc import abstractmethod, ABCMeta
from .drawing import draw_rect, draw_box, in_bbox

RGB_COLORS = {'white': (255, 255, 255),
              'gray': (128, 128, 128),
              'dark_gray': (20, 20, 20)}


def _rgb_to_bgra(rgb, a):
    return rgb + (a,)


class Grid(metaclass=ABCMeta):
    DEFAULT_DRAW_PROPS = {'line_thicknesses': {'thick': 3,
                                               'medium': 2,
                                               'thin': 1},
                          'font': cv2.FONT_HERSHEY_SIMPLEX,
                          'title_font': cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                          'title_thickness': 2,
                          'tick_font_scale': 0.4,
                          'axis_font_scale': 0.6,
                          'title_font_scale': 2,
                          'cursor_font_scale': .8,
                          'draw_bounding_box': True,
                          'tick_string': "%.2f",
                          'mouse_string': None,  # "(%.2f, %.2f)",
                          'marker_string': "(%.2f, %.2f)",
                          'crosshair_length': 20,
                          'tick_length': 20,
                          'user_marker': True,
                          'marker_dot_size': 2,
                          'marker_rad': (10, 15),  # inner, outer
                          'show_ticks': (True, True)}

    DEFAULT_COLORS_BGRA = {'bkg': _rgb_to_bgra(RGB_COLORS['dark_gray'], 255),
                           'heavy': _rgb_to_bgra(RGB_COLORS['white'], 255),
                           'light': _rgb_to_bgra(RGB_COLORS['gray'], 255),
                           'title': _rgb_to_bgra(RGB_COLORS['gray'], 255), }
    TITLE_OPACITY = 0.33

    HOTKEYS = [{'range_up': [ord(']')],  # param 1 adjust
                'range_down': [ord('[')]},
               {'range_up': [ord('0'), ord('o')],  # param 0 adjust ('o' just in case')
                'range_down': [ord('p')]}]

    def __init__(self, bbox,
                 param_ranges=((0., 1.), (0., 1.)),
                 init_values=(None, None),
                 colors=None,
                 draw_props=None,
                 expansion_speed=1.1,
                 title=None,
                 axis_labels=('x', 'y'),
                 minor_ticks=True,
                 minor_unlabeled_ticks=True,
                 adjustability=(True, True)):
        """
        Initialize a new grid.
        :param bbox: dict with 'top','bottom','left','right', where in image is the grid region
        :param param_ranges:  initial range of grid params
        :param colors:  Dict with RGB or RGBA values, {'heavy','light','bkg','title'}
        :param draw_props:  dict with info for drawing (see DEFAULT_DRAW_PROPS)
        :param expansion_speed:  How fast to grow/shrink when user wants to reshape grid
        :param adjustability: (horizontal, vertical)
        """
        self._title = None
        self._adj = adjustability
        # use defaults where not specified
        self._props = Grid.DEFAULT_DRAW_PROPS.copy()
        self._props.update({} if draw_props is None else draw_props)
        self._title = title
        self._axis_labels = axis_labels
        self._param_ranges = np.array(param_ranges)
        self._param_spans = self._param_ranges[:, 1] - self._param_ranges[:, 0]

        self._minors, self._minors_unlabeled = minor_ticks, minor_unlabeled_ticks
        self._bbox = bbox
        self._image_offset = np.array([self._bbox['left'], self._bbox['top']])
        self._size = np.array([bbox['right'] - bbox['left'], bbox['bottom'] - bbox['top']])

        self._set_title_positions()
        self._axis_label_pos = self._get_axis_label_positions()
        self._colors = Grid.DEFAULT_COLORS_BGRA.copy()
        self._colors.update({} if colors is None else colors)

        self._x = expansion_speed

        self._mouse_pos = None
        grid_mean = np.mean(self._param_ranges, axis=1)
        marker_pos_grid = [grid_mean[0] if init_values[0] is None else init_values[0],
                           grid_mean[1] if init_values[1] is None else init_values[1]]
        self._marker_pos_grid = tuple(marker_pos_grid)  # self.grid_coords_to_pixels(marker_pos_grid)

        self._dragging_marker = False

        self._calc_ticks()

    def mouse(self, event, x, y, flags, param):
        """
        Mouse update function, CV2 style.
        Determine local unit coordinates.
        return True if values change
        """

        self._mouse_pos = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            if in_bbox(self._bbox, (x, y)):
                self._dragging_marker = True

        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging_marker = False

        if self._dragging_marker:
            pos = self.pixel_to_grid_coords(self._mouse_pos)
            self._marker_pos_grid = min(max(self._param_ranges[0][0], pos[0]), self._param_ranges[0][1]), \
                                    min(max(self._param_ranges[1][0], pos[1]), self._param_ranges[1][1])

            return True
        return False

    def keyboard(self, k):
        """
        Check for range adjustments, return new ranges if they changed, else None
        """
        rv = None
        for param_i, hotkey in enumerate(Grid.HOTKEYS):
            if not self._adj[param_i]:
                continue
            if k & 0xff in hotkey['range_up']:
                self.set_param_max(max_val=self._param_ranges[param_i, 1] * self._x, param_ind=param_i)
                rv = self._param_ranges
            elif k & 0xff in hotkey['range_down']:
                self.set_param_max(max_val=self._param_ranges[param_i, 1] / self._x, param_ind=param_i)
                rv = self._param_ranges
        return rv

    def move_marker(self, values):
        self._marker_pos_grid = values

    def set_param_max(self, max_val, param_ind):
        if max_val <= self._param_ranges[param_ind, 0]:
            raise Exception("Tried to set param max below param min.")

        self._param_ranges[param_ind][1] = max_val
        self._param_spans = self._param_ranges[:, 1] - self._param_ranges[:, 0]
        self._calc_ticks()

    @abstractmethod
    def _calc_ticks(self):
        """
        set tick mark locations, text, text sizes
        """
        pass

    def get_values(self):
        """
        Marker, if in use, else mouse.
        """
        if self._props['user_marker']:
            pos = self._marker_pos_grid
        else:
            pos = self.pixel_to_grid_coords(self._mouse_pos)

        if pos[0] is None or pos[1] is None:
            return None, None
        return pos

    @abstractmethod
    def pixel_to_grid_coords(self, xy_pos):
        """
        Given the grid image coordinates, what is the grid position.
        """
        pass

    @abstractmethod
    def grid_coords_to_pixels(self, xy):
        """
        Given the grid  coords, what are the image coords
        """
        pass

    @abstractmethod
    def draw(self, image):
        """
        draw grid within self._bbox of image
        """
        pass

    def draw_marker(self, image):
        if self._props['user_marker']:

            m_rad = self._props['marker_rad']

            marker_pos = self.grid_coords_to_pixels(self._marker_pos_grid)
            c = int(marker_pos[0]), int(marker_pos[1])
            box_left = max(c[0] - m_rad[0], self._bbox['left'])
            box_right = min(c[0] + m_rad[0], self._bbox['right'])
            box_top = max(c[1] - m_rad[0], self._bbox['top'])
            box_bottom = min(c[1] + m_rad[0], self._bbox['bottom'])
            w = box_right - box_left
            h = box_bottom - box_top

            draw_rect(image, box_left, box_top, w, h, self._colors['light'])

            cross_left = max(c[0] - m_rad[1], self._bbox['left'])
            cross_right = min(c[0] + m_rad[1], self._bbox['right'])
            cross_top = max(c[1] - m_rad[1], self._bbox['top'])
            cross_bottom = min(c[1] + m_rad[1], self._bbox['bottom'])
            w = cross_right - cross_left
            h = cross_bottom - cross_top
            cross_thickness = 1
            # h
            draw_rect(image, cross_left, int(c[1] - cross_thickness / 2), w, cross_thickness, self._colors['heavy'])
            # v
            draw_rect(image, int(c[0] - cross_thickness / 2), cross_top, cross_thickness, h, self._colors['heavy'])
            if c[1] < (self._bbox['top'] + self._bbox['bottom']) / 2:
                y_offset = 30 * self._props['cursor_font_scale']
            else:
                y_offset = -30 * self._props['cursor_font_scale']
            text_xy = c[0], int(c[1] + y_offset)

            string = self._props['marker_string'] % tuple(self._marker_pos_grid)
            self._write_at_coords(image, string, text_xy, self._colors['heavy'])

    @abstractmethod
    def _get_axis_label_positions(self):
        """
        :returns: (x,y) for param 0, (x,y) for param 1 (will be within bbox)
        """
        pass

    def _set_title_positions(self):

        self._y_center = int(self._bbox['top'] + self._props['tick_length'])
        self._x_center = int(self._bbox['left'] + self._size[0] / 2)

        title = self._title if self._title is not None else ""
        (width, height), baseline = cv2.getTextSize(title,
                                                    self._props['title_font'],
                                                    self._props['title_font_scale'],
                                                    thickness=self._props['title_thickness'])
        self._title_text_pos = self._x_center - int(width / 2), \
                               self._y_center + height

    def _draw_title(self, image):
        if self._title is not None:
            cv2.putText(image, self._title, self._title_text_pos, self._props['title_font'],
                        self._props['title_font_scale'],
                        self._colors['title'], self._props['title_thickness'], cv2.LINE_AA)

            # image[self._y_center, self._x_center, :3] = 255
            # image[self._title_text_pos[1], self._title_text_pos[0], 2] = 255

    def draw_cursor(self, image):

        if self._mouse_pos is None:
            return

        if self._props['mouse_string'] is not None and self._mouse_pos is not None and not self._dragging_marker:
            # text cursor
            values = self.pixel_to_grid_coords(self._mouse_pos)
            string = self._props['mouse_string'] % tuple(values)

            self._write_at_coords(image, string, self._mouse_pos, color=self._colors['heavy'])
        else:
            # crosshair
            xy = self._mouse_pos
            l = int(self._props['crosshair_length'] / 2)
            if self._bbox['left'] + l <= xy[0] < self._bbox['right'] - l and \
                    self._bbox['top'] + l <= xy[1] < self._bbox['bottom'] - l:
                x_px, y_px = self._mouse_pos
                draw_rect(image, x_px, y_px - l, 1, l * 2, self._colors['heavy'])
                draw_rect(image, x_px - l, y_px, l * 2, 1, self._colors['heavy'])

    def _write_at_coords(self, image, string, xy, color):
        """
        Write the coords, at the coords
        """
        (width, height), baseline = cv2.getTextSize(string, self._props['font'], self._props['cursor_font_scale'],
                                                    thickness=1)
        ascender = int(baseline / 2) + 1
        text_y = xy[1] + int(height / 2)  # center text
        text_x = xy[0] - int(width / 2)

        # don't write outside box
        if xy[0] > self._bbox['right'] - width / 2:
            text_x = self._bbox['right'] - width
        if xy[0] < self._bbox['left'] + width / 2:
            text_x = self._bbox['left']
        if xy[1] > self._bbox['bottom'] - height / 2 - baseline:
            text_y = self._bbox['bottom'] - baseline
        if xy[1] < self._bbox['top'] + height / 2 + ascender:
            text_y = self._bbox['top'] + height + ascender

        cv2.putText(image, string, (text_x, text_y), self._props['font'], self._props['cursor_font_scale'],
                    self._colors['heavy'], 1, cv2.LINE_AA)

    def _draw_base_image(self, image):
        """
        helper to draw common elements
        """
        # background
        draw_rect(image, self._bbox['left'], self._bbox['top'],
                  self._size[0], self._size[1],
                  self._colors['bkg'])
        # box
        if self._props['draw_bounding_box']:
            draw_box(image, self._bbox, thickness=self._props['line_thicknesses']['medium'],
                     color=self._colors['heavy'])
        self._draw_title(image)
        self._draw_axis_labels(image)

    def _draw_axis_labels(self, image):
        if self._axis_labels is not None and self._axis_labels[0] is not None:
            cv2.putText(image, self._axis_labels[0], self._axis_label_pos[0], self._props['font'],
                        self._props['axis_font_scale'],
                        self._colors['heavy'], 1, cv2.LINE_AA)
        if self._axis_labels is not None and self._axis_labels[1] is not None:
            cv2.putText(image, self._axis_labels[1], self._axis_label_pos[1], self._props['font'],
                        self._props['axis_font_scale'],
                        self._colors['heavy'], 1, cv2.LINE_AA)


class CartesianGrid(Grid):
    def __init__(self, bbox, **kwargs):
        super(CartesianGrid, self).__init__(bbox, **kwargs)

    def pixel_to_grid_coords(self, xy_pos):
        pos_rel = (np.array(xy_pos) - self._image_offset) / self._size
        pos_rel[1] = 1.0 - pos_rel[1]  # flip y

        return pos_rel * self._param_spans + self._param_ranges[:, 0]

    def grid_coords_to_pixels(self, xy):
        if xy[0] is None or xy[1] is None:
            return None, None
        pos_rel = (np.array(xy) - self._param_ranges[:, 0].reshape(-1)) / self._param_spans
        pos_rel[1] = 1.0 - pos_rel[1]  # flip y
        return (pos_rel * self._size + self._image_offset).astype(np.int64)

    def draw(self, image):
        """
        Draw grid.
        """
        # axes & labels
        self._draw_base_image(image)
        tick_font_scale = self._props['tick_font_scale']
        font = self._props['font']

        # axis ticks
        def _draw_ticks(ticks):
            for tick in ticks:
                color = tick['color']
                draw_rect(image, color=color, **tick['draw_args_a'])
                draw_rect(image, color=color, **tick['draw_args_b'])
                if tick['string'] is not None:
                    cv2.putText(image, tick['string'], tick['text_xy'],
                                self._props['font'], tick_font_scale, color, 1, cv2.LINE_AA)

        for axis in (0, 1):
            if self._props['show_ticks'][axis]:
                _draw_ticks(self._param_ticks[axis]['major'])
                _draw_ticks(self._param_ticks[axis]['minor_labeled'])
                _draw_ticks(self._param_ticks[axis]['minor_unlabeled'])

        # cursor
        self.draw_marker(image)
        self.draw_cursor(image)

    def _get_axis_label_positions(self):
        # H-axis
        h_indent = 30
        label = self._axis_labels[0] if self._axis_labels is not None and self._axis_labels[0] is not None else ""
        (width, height), baseline = cv2.getTextSize(label,
                                                    self._props['font'],
                                                    self._props['axis_font_scale'],
                                                    thickness=1)
        h_label_pos = int(self._bbox['right'] - width - h_indent), \
                      int(self._bbox['bottom'] - baseline * 1.5)
        # V-axis
        h_indent = 10
        v_indent = 20
        label = self._axis_labels[1] if self._axis_labels is not None and self._axis_labels[1] is not None else ""
        (width, height), baseline = cv2.getTextSize(label,
                                                    self._props['font'],
                                                    self._props['axis_font_scale'],
                                                    thickness=1)
        v_label_pos = int(self._bbox['left'] + h_indent), \
                      int(self._bbox['top'] + height + baseline + v_indent)
        return h_label_pos, v_label_pos

    def _calc_ticks(self):
        """
        Get good places for tick marks:
            1. near round numbers
            2. spanning whole range, but not too close to edges
        """

        high_margin_frac = 0.1  # don't put tics within this fraction of the high end of the range
        low_margin_frac = 0.01  # don't put tics within this fraction of zero

        # calc  step size, at least 10 ticks of minor order
        def _calc(low, high, is_vertical):
            span = high - low
            margin = {'high': (high_margin_frac * span),
                      'low': (low_margin_frac * span)}
            span_order = np.log10(span)

            major_order = np.floor(span_order)
            minor_order = major_order - 1

            major_step = 10. ** major_order
            minor_step = 10. ** minor_order

            # find where scales start (can be out of range)
            major_tick_start = np.floor(low / major_step)
            labeled_minor_tick_start = major_tick_start - 0.5 * major_step
            unlabeled_minor_tick_start = np.floor(low / minor_step)

            n_major = np.floor(span / major_step) * 2  # overkill
            n_minor = np.floor(span / minor_step) * 2

            major_tick_locs = major_tick_start + np.arange(0, n_major) * major_step
            unlabeled_minor_tick_locs = unlabeled_minor_tick_start + np.arange(0, n_minor) * minor_step
            labeled_minor_tick_locs = labeled_minor_tick_start + np.arange(0, n_major + 1) * major_step
            if not self._minors:
                labeled_minor_tick_locs = []

            unlabeled_minor_tick_locs = [m for m in unlabeled_minor_tick_locs
                                         if m not in major_tick_locs and m not in labeled_minor_tick_locs]

            tick_thickness = self._props['line_thicknesses']['thin']

            def _get_text_and_sizes(value, tick_length, color, labeled=True):
                if labeled:
                    string = self._props['tick_string'] % (value,)
                    size = cv2.getTextSize(string, self._props['font'],
                                           self._props['tick_font_scale'],
                                           thickness=1) if labeled else None
                else:
                    string = None
                    size = (0, 0), 0
                # for text positions
                y_bottom = self._bbox['bottom'] - tick_length - 4 - size[1]
                x_left = self._bbox['left'] + tick_length + 4

                if is_vertical:
                    # drawing on left & right, writing on left
                    y_rel = 1.0 - (value - self._param_ranges[1, 0]) / self._param_spans[1]  # flip y
                    y = int(self._bbox['top'] + y_rel * self._size[1])
                    draw_args_a = {'left': self._bbox['left'], 'width': tick_length,
                                   'top': y, 'height': tick_thickness}
                    draw_args_b = {'left': self._bbox['right'] - tick_length, 'width': tick_length,
                                   'top': y, 'height': tick_thickness}
                    text_xy = x_left, y + int(size[0][1] / 2) if string is not None else None
                else:
                    # drawing on top & bottom, writing on bottom
                    x_rel = (value - self._param_ranges[0, 0]) / self._param_spans[0]
                    x = int(self._bbox['left'] + x_rel * self._size[0])
                    draw_args_a = {'left': x, 'width': tick_thickness,
                                   'top': self._bbox['top'], 'height': tick_length}
                    draw_args_b = {'left': x, 'width': tick_thickness,
                                   'top': self._bbox['bottom'] - tick_length, 'height': tick_length}
                    text_xy = x - int(size[0][0] / 2), y_bottom if string is not None else None

                return {'value': value,
                        'text_xy': text_xy,
                        'draw_args_a': draw_args_a,
                        'draw_args_b': draw_args_b,
                        'string': string,
                        'size': size,
                        'color': color}

            def _in_range(value):
                return low + margin['low'] <= value <= high - margin['high']

            major_labels = [_get_text_and_sizes(tick_value, self._props['tick_length'], self._colors['heavy'], True)
                            for tick_value in major_tick_locs if _in_range(tick_value)]
            minor_labels = [
                _get_text_and_sizes(tick_value, int(self._props['tick_length'] * 2. / 3.), self._colors['light'], True)
                for tick_value in labeled_minor_tick_locs if _in_range(tick_value)] if self._minors else []
            unlabeled_minor_ticks = [_get_text_and_sizes(tick_value,
                                                         int(self._props['tick_length'] / 3.),
                                                         self._colors['light'],
                                                         False)
                                     for tick_value in unlabeled_minor_tick_locs if
                                     _in_range(tick_value)] if self._minors_unlabeled else []

            return {'major': major_labels,
                    'minor_labeled': minor_labels,
                    'minor_unlabeled': unlabeled_minor_ticks}

        self._param_ticks = [_calc(p_range[0], p_range[1], vert)
                             for p_range, vert in zip([self._param_ranges[0, :],
                                                       self._param_ranges[1, :]],
                                                      (False, True))]


def _test_conversions(grid, size):
    for _ in range(100):
        xy = np.random.rand(2) * size
        xy_grid = grid.pixel_to_grid_coords(xy)
        xy_prime = grid.grid_coords_to_pixels(xy_grid)
        if int(xy[0]) != xy_prime[0] or int(xy[1]) != xy_prime[1]:
            raise Exception("Conversion wrong:  %s  ->  %s  ->  %s" % (xy, xy_grid, xy_prime))


def grid_sandbox():
    shape = (700, 1000, 4)
    blank = np.zeros(shape, np.uint8)
    bbox = {'top': 10, 'bottom': blank.shape[0] - 10, 'left': 10, 'right': blank.shape[1] - 10}
    grid = CartesianGrid(bbox, param_ranges=((0.0, 1.5), (0.0, 78.3567456735)), title='Grid')
    window_name = "Grid sandbox"
    _test_conversions(grid, (shape[1], shape[0]))

    def _mouse(event, x, y, flags, param):
        grid.mouse(event, x, y, flags, param)
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Click:  %s" % (grid.get_values(),))
        elif event == cv2.EVENT_LBUTTONUP:
            print("Un-slick:  %s" % (grid.get_values(),))

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse)
    draw_times = []
    frame_count = 0
    while True:
        frame = blank.copy()

        t_start = time.perf_counter()
        grid.draw(frame)
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
