import cv2
from abc import abstractmethod, ABCMeta
import numpy as np


def _scale_unit_coords(coords, shape):
    """
    Get absolute coordinates, scaled to given image.
    Do not preserve aspect ratio.
    Flip y-axis.

    :param shape:  N x 2, xy coords in unit square
    :param shape:  HxWxC, i.e. shape of image
    :return:  N x 2 float array, scaled
    """

    coords = np.array(coords).reshape(-1, 2)

    scale = np.array([shape[1], shape[0]]).reshape(1, 2)
    flipped = np.hstack((coords[:, 0].reshape(-1, 1),
                         1.0 - coords[:, 1].reshape(-1, 1)))
    print(coords.shape, flipped.shape)
    return flipped * scale


class DrawableComponent(object):
    """
    Draw a small element of an image.
    Line, shape, etc.

    """
    PRECISION_BITS = 0
    PRECISION_MULT = 2. ** PRECISION_BITS
    LINE_TYPE = cv2.LINE_AA

    def __init__(self, coords, draw_color=None, is_closed=False, fill_color=None, thickness=1, name=""):
        """
        Roughly, arguments of cv2.draw_polylines.

        Define in the unit square, but will be rendered scaled to an arbitrary shape.  (i.e. aspect ratio is
        part of the figure rendering, not the component definition)

        :param coords:  N x 2 array, outline of shape,  relative to [0, 1] x [0, 1], origin at bottom right.
        :param draw_color:    1, 3 or 4 element (rgb, etc.), or NONE if not drawing border/line
            default to be used if alt is not specified at render time.
        :param is_closed:  Connect first and last coords?  (ignored, set to True if fill_color provided)
        :param fill_color:  same, or NONE if not drawing filled.  (same default behavior as draw_color)
        :param thickness:  line thickness, or None if not drawing border/line
        """
        self._coords = np.array(coords)
        self._is_closed = is_closed if fill_color is None else True
        self._thickness = thickness
        self._draw_color, self._fill_color = draw_color, fill_color
        self._name = name

        if not self._is_closed:
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

        scaled_coords = _scale_unit_coords(self._coords, image.shape)

        coords = (scaled_coords * DrawableComponent.PRECISION_MULT).astype(np.int32)
        pts = coords.reshape((-1, 1, 2))

        if self._draw_interior:
            line_type = DrawableComponent.LINE_TYPE if not self._draw_line else cv2.LINE_8
            cv2.fillPoly(image, [pts], fill_color[:3],
                         line_type, DrawableComponent.PRECISION_BITS)

        if self._draw_line:
            cv2.polylines(image, [pts], self._is_closed, draw_color, self._thickness,
                          DrawableComponent.LINE_TYPE, DrawableComponent.PRECISION_BITS)

    def get_name(self):
        return self._name


class Artist(object, metaclass=ABCMeta):
    """
    Generic  class for small images.

    Use by inheriting from this class or one of its subclasses.
    """

    def __init__(self, bkg_color=(255, 255, 255, 0)):
        """
        Initialize an icon/cursor/sprite, etc.
        """
        self._bkg_color = np.array(bkg_color, dtype=np.uint8).reshape((1, 1, -1))
        self._components = self._get_components()

    @abstractmethod
    def _get_components(self):
        """
        :return: list of DrawableComponent objects.
        Set figure definitions here.   (see test_figure_drawing.py for an example)
        """
        pass

    def _get_blank(self, shape):
        return np.zeros((shape[0], shape[1], len(self._bkg_color)), dtype=np.uint8) + self._bkg_color

    def make(self, shape, color_substitutions=None):
        """
        Render the figure into a new image.
        :param shape:  image shape (H x W x Colors)
        :param color_substitutions:  dict(component_name: {fill_color=new_color/None,
                                                           draw_color=new_color/None},
                                          ...)
        :return:  image,
        (x,y) ctrl point
        """
        img = self._get_blank(shape)
        for comp in self._components:
            color_subs = getattr(color_substitutions, comp.get_name(), dict(fill_color=None, draw_color=None))
            comp.draw(img, **color_subs)

        return img


class SpriteArtist(Artist):
    """
    Borderless figure with single "control point".
    """

    def __init__(self, trim_edges=True, *args, **kwargs):
        super(SpriteArtist, self).__init__(*args, **kwargs)
        self._control_xy = np.array(self._get_control_xy()).reshape(-1)
        self._trim = trim_edges

    @abstractmethod
    def _get_control_xy(self):
        """
        Get the reference point of the figure, wrt unit square.
        :return:  (x, y) in [0,1]x[0,1]
        """

    def make(self, shape, color_substitutions=None):
        """
        Same params as Artist.make()
        :return:  image,j
          control (control point x,y within that image)
        """
        img = super(SpriteArtist, self).make(shape, color_substitutions)
        control_xy = _scale_unit_coords(self._control_xy, img.shape)

        if self._trim:
            img, offset = self._prune_image_sides(img)
            control_xy -= offset

        return img, control_xy.reshape(-1)

    def _prune_image_sides(self, image):
        """
        Remove rows/columns at edges of image if they match background color.
        :param image:  image to prune HxWxd or HxWxd, uint8
        :bkg color: d-element uint8
        :returns:  hxwxd image, where h<=H and w<=W, and each of the four edge-rows of pixels is not just background color.
             (n_left_cols_removed, n_top_rows_removed), i.e. the x,y offset for all (pixel) coords
        """
        n_c_channels = self._bkg_color.size
        bkg = np.array(self._bkg_color, dtype=np.uint8)
        # print(image[:,:,].shape, bkg[None, None,:].shape)
        matching = image == bkg  # places where image matches color

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
