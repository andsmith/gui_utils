"""
Dialog box for picking options, either list of text items, or list of images/icons.
"""
import tkinter as tk
from tkinter import ttk
import logging
import numpy as np
from PIL import Image, ImageTk
from copy import deepcopy


class ChooseItemDialog(object):
    """
    Class for dialog box.
    """
    CONFIG = {'label_font': ("Arial", 12), 'button_font': ("Arial", 12)}

    def __init__(self, prompt="Select one of the following:",
                 button_text="Continue...", label_params=None, item_params=None, button_params=None,
                 grid_spacing=None, initial_selection=0, window_title="Make a selection."):

        """"
        :param prompt:  Label above items
        :param button_text:  text for selection button
        :param item_params:  dict with kwargs for tk.RadioButton
        :param button_params:  dict with kwargs for tk.Button
        :param initial_selection:  Which item starts selected
        :param grid_spacing: {label: {'pady': 10}, items: {...}, button: {...}}  (unused for text mode)
        :param window_title: at top of window.
        :return: index of label item clicked or None for window closed
        """
        self._window_title = window_title
        self._initial_selection = initial_selection
        self._prompt = prompt
        self._button_text = button_text
        self._item_params = ChooseItemDialog.update_default_params({}, item_params)  # add more defaults in {}
        self._button_params = ChooseItemDialog.update_default_params({}, button_params)
        self._label_params = ChooseItemDialog.update_default_params({}, label_params)
        self._grid_spacing = {'items': {'padx': 7, 'pady': 7}, 'label': {'pady': 10},
                              'button': {'pady': 10}} if grid_spacing is None else grid_spacing

    @staticmethod
    def update_default_params(defaults, updates):
        if updates is not None:
            defaults = deepcopy(defaults)
            defaults.update(updates)
        return defaults

    def _tk_init(self):
        self._root = tk.Tk()
        self._root.title(self._window_title)
        self._content = ttk.Frame(self._root)

        self._var = tk.IntVar()
        self._var.set(0)

    def _finish(self):
        logging.info("Icon selected:  %s" % (self._var.get(),))
        self._root.destroy()

    def _click(self):
        pass
        logging.info("Icon clicked:  %s" % (self._var.get(),))

    def ask_icons(self, icons):
        """
        Ask user to pick one of the icons.
        :param icons: grid (list of lists) of images to pick from. If there are n icons, and n< n_rows *n_cols,
            only the last row should have fewer than the others.
        """
        self._tk_init()

        # so grid "stretches" when window is resided
        self._root.rowconfigure(0, weight=1)
        self._root.columnconfigure(0, weight=1)
        self._content.grid(column=0, row=0, sticky="news")

        icon_val = 0
        n_rows, n_cols = len(icons), len(icons[0])

        images = [[ImageTk.PhotoImage(Image.fromarray(ic)) for ic in icon_row] for icon_row in icons]
        label = tk.Label(self._content, text=self._prompt, font=ChooseItemDialog.CONFIG['label_font'],
                         **self._label_params)
        label.grid(row=0, columnspan=n_cols, **self._grid_spacing['label'])
        button = tk.Button(self._content, text=self._button_text, command=self._finish,
                           font=ChooseItemDialog.CONFIG['button_font'], **self._button_params)
        button.grid(row=n_rows + 1, columnspan=n_cols, **self._grid_spacing['button'])
        for row, image_row in enumerate(images):
            for col, im in enumerate(image_row):
                rb = tk.Radiobutton(self._content, indicatoron=0,
                                    variable=self._var,
                                    value=icon_val,
                                    image=im, **self._item_params)
                rb.grid(column=col, row=row + 1, **self._grid_spacing['items'])
                icon_val += 1

        self._content.columnconfigure(tuple(range(n_cols)), weight=1)
        self._content.rowconfigure(tuple(range(n_rows + 2)), weight=1)

        self._root.mainloop()
        return self._var.get()

    def ask_text(self, choices):
        """
        Ask user to pick one of the text choices.
        :param choices: list of strings.
        """
        self._tk_init()

        label = tk.Label(self._root, text=self._prompt, **self._label_params)
        label.pack(expand=True, **self._grid_spacing['label'])

        for i, label in enumerate(choices):
            rb = tk.Radiobutton(self._root,
                                text=label,
                                variable=self._var,
                                value=i, command=self._click,
                                **self._item_params)
            rb.pack(anchor=tk.W, expand=True, **self._grid_spacing['items'])

        button = tk.Button(self._root,
                           text=self._button_text,
                           command=self._finish,
                           **self._button_params)
        button.pack(expand=True, **self._grid_spacing['button'])

        self._root.mainloop()
        return self._var.get()


def make_test_image(shape, color_range=(0, 255), n_rects=10):
    if len(shape) == 2:
        shape = (shape[0], shape[1], 3)
    n_channels = shape[2]
    img = np.zeros(shape).astype(np.uint8) + 254
    s = 15
    for x in range(n_rects):
        io = np.random.randint(0, shape[0] - s)
        jo = np.random.randint(0, shape[1] - s)
        color = np.random.randint(color_range[0], color_range[1], n_channels)
        if n_channels == 4:
            color[3] = 255
        img[io:io + s, jo:jo + s, :] = color
    return img


def _make_test_icons(n):
    imgs = []
    icon_res = 60, 60
    color_ranges = np.linspace(0, 254, n + 1)

    for ind in range(n):
        img = make_test_image(icon_res, (color_ranges[ind], color_ranges[ind + 1]))
        imgs.append(img)

    return imgs


def choose_item_text(choices=None, prompt="> "):
    """
    No GUI elements, just text mode.
    """
    selection = -1
    while True:
        failed = False
        print("\nSelect one of the following (or 0 for None):")
        for i, choice in enumerate(choices):
            print("\t%i) %s" % (i + 1, choice))
        try:
            selection = int(input(prompt)) - 1
        except:
            failed = True
        if selection < -1 or selection >= len(choices):
            failed = True
        if failed:
            print("\n\nPlease enter a valid choice!!!\n")
            continue
        break
    if selection is None or selection == -1:
        return None
    return selection


def _test_icon_picker():  # test requires user interaction
    n_icons = 6
    icons = _make_test_icons(n_icons)
    icon_grid = [icons[:3], icons[3:5]]
    logging.info("Testing selection using icons")
    choice = ChooseItemDialog().ask_icons(icons=icon_grid)
    print("User chose:  %s" % (choice,))


def _test_picker():  # test requires user interaction
    choices = ['item 1', 'item 2 is longer', 'item 3']
    logging.info("Testing GUI selection dialog")
    choice = ChooseItemDialog().ask_text(choices=choices)
    logging.info("User chose:  %s" % (choice,))
    logging.info("Testing command line selection")
    choice = choose_item_text(choices=choices)
    logging.info("User chose:  %s" % (choice,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    _test_picker()
    _test_icon_picker()
