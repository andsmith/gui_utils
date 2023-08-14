from abc import abstractmethod, ABCMeta

    class PencilArtist(Artist):

        def _draw(self, s, color=None):
            pencil, tip = self.get_pencil_figure_params()
            icon, (col_offset, row_offset) = draw_centered_shape(s, pencil, bkg_color=self._bkg_color, flip_bgr=True)

            icon_tip_pos = (int(tip[0] * s) - col_offset,
                            int((1. - tip[1]) * s) - row_offset)

            return icon, icon_tip_pos

        def get_pencil_figure_params(self):
            """
            For drawing other icons that have a pencil in them.
            """

            # counterclockwise from tip
            pencil_band_length = LAYOUT['pencil']['band_length']
            pencil_width = LAYOUT['pencil']['width']
            pencil_length = LAYOUT['pencil']['length']
            pencil_sharpness = pencil_width * LAYOUT['pencil']['tip_sharpness_ratio']
            eraser_length = 0.05

            tip = np.array([0.05, 0.05])
            rim_pos = tip + pencil_sharpness
            lower_tip_corner = rim_pos + np.array([1., -1.]) * pencil_width / 2
            eraser_pos = rim_pos + pencil_length
            lower_eraser_corner = eraser_pos + np.array([1., -1.]) * pencil_width / 2
            lower_band_corner = lower_eraser_corner - np.array([1., 1.]) * pencil_band_length

            end_pos = eraser_pos + eraser_length
            lower_end_corner = end_pos + np.array([1., -1.]) * pencil_width / 2
            upper_band_corner = lower_band_corner[::-1]
            upper_end_corner = lower_end_corner[::-1]
            upper_eraser_corner = lower_eraser_corner[::-1]
            upper_tip_corner = lower_tip_corner[::-1]

            colors = LAYOUT['pencil']['colors']
            tip_color = colors['tip']
            pencil = [
                # tip
                {'coords': flip_vertical_rel([tip, lower_tip_corner, upper_tip_corner, tip]),
                 'closed': True,
                 'border_color': add_alpha(colors['outline'], 255),
                 'fill_color': add_alpha(tip_color, 255),
                 'thickness': OUTLINE_THICKNESS},
                # body
                {'coords': flip_vertical_rel(
                    [lower_tip_corner, lower_eraser_corner, upper_eraser_corner, upper_tip_corner]),
                    'closed': True,
                    'border_color': add_alpha(colors['outline'], 255),
                    'fill_color': add_alpha(colors['body'], 255),
                    'thickness': OUTLINE_THICKNESS},

                # line down middle
                {'coords': flip_vertical_rel([rim_pos, eraser_pos]),
                 'closed': False,
                 'border_color': add_alpha(colors['line'], 255),
                 'fill_color': None,
                 'thickness': 1},
                # Eraser
                {'coords': flip_vertical_rel(
                    [lower_eraser_corner, lower_end_corner, upper_end_corner, upper_eraser_corner]),
                    'closed': True,
                    'border_color': add_alpha(colors['outline'], 255),
                    'fill_color': add_alpha(colors['eraser'], 255),
                    'thickness': OUTLINE_THICKNESS},
                {'coords': flip_vertical_rel(
                    [lower_eraser_corner, lower_band_corner, upper_band_corner, upper_eraser_corner]),
                    'closed': True,
                    'border_color': add_alpha(colors['outline'], 255),
                    'fill_color': add_alpha(colors['band'], 255),
                    'thickness': OUTLINE_THICKNESS}
            ]
            return pencil, tip
