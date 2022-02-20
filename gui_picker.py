import tkinter as tk
import logging


def choose_item_dialog(labels, prompt="Please select one\nof the following:",
                       button_text="Continue...", label_params=None, item_params=None):
    """
    Open a dialog box with some text, radio buttons, and continue button.
    :param labels:  text for radio buttons
    :param prompt:  Label above items
    :param button_text:  text for selection button
    :param label_params:  dict with kwargs to tk.Label
    :param item_params:  dict with kwargs for tk.RadioButton
    :return: index of label item clicked or None for window closed
    """
    indent_px = 20

    # update defaults
    lp = {'padx': indent_px, 'pady': 5}  # defaults
    if label_params is not None:
        lp.update(label_params)
    label_params = lp

    ip = {'padx': indent_px, 'pady': 0}  # defaults
    if item_params is not None:
        ip.update(item_params)
    item_params = ip

    root = tk.Tk()
    selection = tk.IntVar(value=0)

    # Default is first item
    selection.set(0)
    final_choice = [0]
    finished=[False]

    def show_choice():
        n = selection.get()
        # logging.info("Item %i selected (%s)" % (n, labels[n]))
        final_choice[0] = n

    def finish():
        finished[0]=True
        # logging.info("Selection finalized:  %s" % (final_choice,))
        root.destroy()

    label = tk.Label(root, text=prompt, **label_params)
    label.pack(pady=5)

    for i, label in enumerate(labels):
        rb = tk.Radiobutton(root,
                            text=label,
                            variable=selection,
                            command=show_choice,
                            value=i,
                            **item_params)
        rb.pack(anchor=tk.W)

    button = tk.Button(root,
                       text=button_text,
                       command=finish,
                       padx=indent_px, pady=5)
    button.pack(pady=20)

    root.mainloop()
    if not finished[0]:
        return None
    return final_choice[0]


def _test_picker():  # requires user interaction
    choice = choose_item_dialog(label_params=dict(justify=tk.LEFT),
                                labels=['item 1', 'item 2 is longer', 'item 3'])
    print("User chose:  %s" % (choice,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_picker()
