import cv2
from abc import abstractmethod, ABCMeta
import numpy as np


class DrawableComponent(object):
    """
    Draw a small element of an image.
    Line, shape, etc.

    """
    PRECISION_BITS = 6
    PRECISION_MULT = 2. ** PRECISION_BITS
    LINE_TYPE = cv2.LINE_AA

    def __init__(self, coords, draw_color, is_closed=False, fill_color=None, thickness=1, name=""):
        """
        Roughly, arguments of cv2.draw_polylines.

        Define in the unit square, but will be rendered scaled to an arbitrary shape.  (i.e. aspect ratio is
        part of the figure rendering, not the component definition)

        :param coords:  N x 2 array, outline of shape,  relative to [0, 1] x [0, 1], origin at bottom right.
        :param draw_color:    1, 3 or 4 element (rgb, etc.), or NONE if not drawing border/line
            default to be used if alt is not specified at render time.
        :param is_closed:  Connect first and last coords?
        :param fill_color:  same, or NONE if not drawing filled.  (same default behavior as draw_color)
        :param thickness:  line thickness, or None if not drawing border/line
        """
        self._coords = coords
        self._is_closed = is_closed
        self._thickness = thickness
        self._draw_color, self._fill_color = draw_color, fill_color
        self._name = name

        if not self._is_closed:
            if self._fill_color is not None:
                raise Exception("Fill color provided for non-closed figure.")
            if self._thickness is None:
                raise Exception("Line thickness must be given for non-closed figures")

        x_min, y_min = np.min(self._coords, axis=0)
        x_max, y_max = np.max(self._coords, axis=0)
        if x_min < 0.0 or x_max > 1.0 or y_min < 0.0 or y_max > 1.0:
            raise Exception("Need coordinates within unit square")

        self._draw_line = draw_color is not None
        self._draw_interior = fill_color is not None

        if not self._draw_line and not self._draw_interior:
            raise Exception("Inconsistent border/color/closed arguments")

    def get_scaled_coords(self, shape):
        """
        Unit square coordinates -> image pixel coordinates.
        :param shape:  image H x W
        :return:  pixel_coords, N x 2 array of floats
            pixel_center, 1 x 2 array
        """
        scale = np.array([shape[1], shape[0]]).reshape(1, 2)
        return self._coords * scale

    def draw(self, image, draw_color=None, fill_color=None):
        """
        render to an image
        :param image: draw component on this image
        :param draw_color: Substitute this color for the figure outline
        :param fill_color: Substitute this color for the figure interior (if filled)
        :return:
        """
        draw_color = draw_color if draw_color is not None else self._draw_color
        fill_color = fill_color if fill_color is not None else self._fill_color

        coords = (self.get_scaled_coords(image.shape) * DrawableComponent.PRECISION_MULT).astype(np.int32)
        pts = coords.reshape((-1, 1, 2))

        if self._draw_interior:
            cv2.fillPoly(image, [pts], fill_color[:3],
                         DrawableComponent.LINE_TYPE, DrawableComponent.PRECISION_BITS)

        if self._draw_line:
            cv2.polylines(image, [pts], self._is_closed, draw_color, self._thickness,
                          DrawableComponent.LINE_TYPE, DrawableComponent.PRECISION_BITS)

    def get_name(self):
        return self._name


class Figure(object):
    """
    Generic base class for small images.
    """

    def __init__(self, components, bkg_color=(255, 255, 255, 0)):
        """
        Initialize an icon/cursor/sprite, etc.
        :param components:  list of DrawableComponent objects.
        """
        self._components = components
        self._bkg_color = bkg_color

    def _get_blank(self, shape):
        return np.zeros(shape, dtype=np.uint8) + self._bkg_color.reshape((1, 1, -1))

    def make(self, shape, color_substitutions=None):
        """
        Render the figure into a new image.
        :param shape:  image shape (H x W x Colors)
        :param color_substitutions:  dict(component_name: {fill_color=new_color/None,
                                                           border_color=new_color/None},
                                          ...)
        :return:  image,
        (x,y) center point
        """
        img = self._get_blank(shape)
        for comp in self._components:
            color_subs = getattr(color_substitutions, comp.get_name(), dict(fill_color=None, border_color=None))
            comp.draw(img, **color_subs)

        return img




class Sprite(Figure):
    """
    Borderless figure with precise "center point".
    """

    def __init__(self, components, center_xy, trim_edges=True, *args, **kwargs):
        super(Sprite, self).__init__(components, *args, **kwargs)
        self._center = center_xy
        self._trim = trim_edges

    def make(self, shape, color_substitutions=None):
        img = super(Sprite, self).make(shape, color_substitutions)
        scale = np.array([shape[1], shape[0]]).reshape(1, 2)

        center_xy = self._center * scale

        if self._trim:
            img, center_offset = self._prune_image_sides(img)
            center_xy -= np.array(center_offset).reshape(1,2)

        return img, center_xy

    def _prune_image_sides(self, image):
        """
        Remove rows/columns at edges of image if they match background color.
        :param image:  image to prune HxWxd or HxWxd, uint8
        :bkg color: d-element uint8
        :returns:  hxwxd image, where h<=H and w<=W, and each of the four edge-rows of pixels is not just background color.
             (n_left_cols_removed, n_top_rows_removed), i.e. the x,y offset for all (pixel) coords
        """
        n_c_channels = len(self._bkg_color)
        bkg = np.array(self._bkg_color, dtype=np.uint8)
        matching = image[:, :, ] == bkg[None, None, :]  # places where image matches color

        matching = np.sum(matching, axis=2) == n_c_channels  # match all channels (incl. alpha)

        rows_to_clear = np.sum(matching, axis=1) == image.shape[0]
        cols_to_clear = np.sum(matching, axis=0) == image.shape[1]

        first_non_bkg_column = np.where(np.logical_not(cols_to_clear))[0][0]
        last_non_bkg_column = image.shape[1] - np.where(np.logical_not(cols_to_clear[::-1]))[0][0]
        first_non_bkg_row = np.where(np.logical_not(rows_to_clear))[0][0]
        last_non_bkg_row = image.shape[0] - np.where(np.logical_not(rows_to_clear[::-1]))[0][0]

        img_pruned = image[first_non_bkg_row:last_non_bkg_row, first_non_bkg_column:last_non_bkg_column, :]

        n_left_cols_removed = first_non_bkg_column
        n_top_rows_removed = first_non_bkg_row

        return img_pruned, (n_left_cols_removed, n_top_rows_removed)
