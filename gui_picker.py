import tkinter as tk
import logging
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from copy import deepcopy


def choose_item_dialog(labels, icons=None, prompt="Please select one\nof the following:",
                       button_text="Continue...", label_params=None, item_params=None, button_params=None,
                       default_selection=0):
    """
    Open a dialog box with some text, radio buttons, and continue button.
    :param labels:  text for radio buttons, list of images to select one of those
    :param icons:  (grid (list of lists) of icons, grid of selected icons), each image WxHx3 uint8
    :param prompt:  Label above items
    :param button_text:  text for selection button
    :param label_params:  dict with kwargs to tk.Label
    :param item_params:  dict with kwargs for tk.RadioButton
    :param button_params:  dict with kwargs for tk.Button
    :param default_selection:  Which item starts selected
    :return: index of label item clicked or None for window closed
    """

    def update_default_params(defaults, updates):
        if updates is not None:
            defaults = deepcopy(defaults)
            defaults.update(updates)
        return defaults

    button_params = update_default_params({}, button_params)

    root = tk.Tk()
    selection = tk.IntVar(value=0)

    # set initial selection
    selection.set(default_selection)
    final_choice = [default_selection]
    finished = [False]

    def show_choice():
        n = selection.get()
        # logging.info("Item %i selected (%s)" % (n, labels[n]))
        final_choice[0] = n

    def finish():
        finished[0] = True
        # logging.info("Selection finalized:  %s" % (final_choice,))
        root.destroy()

    if icons is None:
        # text mode

        indent_px = 20

        # "label" is the text at top
        label_params = update_default_params({'padx': indent_px, 'pady': 5}, label_params)

        # "items" are the option to pick
        item_params = update_default_params({'padx': indent_px, 'pady': 5}, item_params)

        label = tk.Label(root, text=prompt, **label_params)
        label.pack(pady=5, expand=True)

        for i, label in enumerate(labels):
            rb = tk.Radiobutton(root,
                                text=label,
                                variable=selection,
                                command=show_choice,
                                value=i,
                                **item_params)
            rb.pack(anchor=tk.W, expand=True)

        button = tk.Button(root,
                           text=button_text,
                           command=finish,
                           **button_params)
        button.pack(expand=True)

    else:
        # ICON mode
        label_params = update_default_params({}, label_params)
        item_params = update_default_params({'padx': 10, 'pady': 10,}, item_params)

        unsel, sel = icons
        n_cols, n_rows = len(unsel), len(unsel[0])

        label = tk.Label(root, text=prompt)
        label.grid(row=0, column=0, columnspan=n_cols)

        icon_values = 0
        for row in range(len(unsel)):
            for col in range(len(unsel[row])):
                s, us = sel[row][col], unsel[row][col]
                if s is None:
                    break

                unsel_icon = ImageTk.PhotoImage(Image.fromarray(us), name=labels[row][col])
                sel_icon = ImageTk.PhotoImage(Image.fromarray(s))
                rb = tk.Radiobutton(root,
                                    image=unsel_icon,
                                    selectimage=sel_icon,
                                    indicatoron=0,
                                    variable=selection,
                                    command=show_choice,
                                    value=icon_values,
                                    **item_params)
                icon_values += 1
                rb.grid(column=col, row=row + 1)

        button = tk.Button(root,
                           text=button_text,
                           command=finish)
        button.grid(row=n_rows + 1, column=0, columnspan=n_cols)

    root.mainloop()
    if not finished[0]:
        return None
    return final_choice[0]


def _make_test_icons(n):
    imgs = []
    icon_res = 60, 60
    color_ranges = np.linspace(0, 254, n+1)
    for ind in range(n):

        img = np.zeros((icon_res[1], icon_res[0], 3)).astype(np.uint8) + 254
        s = 15
        for x in range(10):
            io = np.random.randint(0, icon_res[1] - s)
            jo = np.random.randint(0, icon_res[0] - s)
            color = np.random.randint(color_ranges[ind], color_ranges[ind+1], 3)
            img[jo:jo + s, io:io + s, :] = color
        imgs.append(img)

    select_width = 6
    selected_imgs = []
    for img in imgs:
        s_img = img * 0 + 200
        s_img[select_width:-select_width, select_width:-select_width, :] = \
            img[select_width:-select_width, select_width:-select_width, :]
        selected_imgs.append(s_img)


    return imgs, selected_imgs


def _test_icon_picker():
    n_icons = 6
    u_icons, s_icons = _make_test_icons(n_icons)
    labels = ["Item %i" % (i + 1,) for i in range(n_icons)]

    # make grids
    unselected = [u_icons[:3], u_icons[3:5]]
    selected = [s_icons[:3], s_icons[3:5]]
    labels = [labels[:3], labels[3:5]]

    choice = choose_item_dialog(labels=labels, icons=(unselected, selected))
    print("User chose:  %s" % (choice,))


def _test_picker():  # requires user interaction
    choice = choose_item_dialog(labels=['item 1', 'item 2 is longer', 'item 3'])
    print("User chose:  %s" % (choice,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    #import ipdb; ipdb.set_trace()
    #_test_picker()
    _test_icon_picker()
