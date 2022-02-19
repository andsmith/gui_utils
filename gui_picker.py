import tkinter as tk
import logging


def choose_item_dialog(prompt, labels, values, button_text="Continue...",
                       label_params=None, item_params=None):
    """
    Open a dialog box with some text, radio buttons, and continue button.
    :param prompt:  text at top
    :param labels:  text for radio buttons
    :param values:  values for radio buttons
    :param button_text:  text for selection button
    :param label_params:  dict with kwargs to tk.Label
    :param item_params:  dict with kwargs for tk.RadioButton
    :return:
    """
    indent_px = 20
    lp = {'padx': indent_px, 'pady': 5}  # defaults
    if label_params is not None:
        lp.update(label_params)
    label_params = lp

    ip = {'padx': indent_px, 'pady': 0}  # defaults
    if item_params is not None:
        ip.update(item_params)
    item_params = ip

    root = tk.Tk()

    selection = tk.IntVar(value=1)
    selection.set(1)  # i+1
    final_choice = [0]

    def show_choice():
        n = selection.get() - 1
        logging.info("Item %i selected (text='%s', value=%i)" % (n, labels[n], values[n]))
        final_choice[0] = n

    def finish():
        logging.info("Selection finalized:  %s" % (final_choice,))
        root.destroy()

    label = tk.Label(root, text=prompt, **label_params)
    label.pack(pady=5)

    for i, (label, value) in enumerate(zip(labels, values)):
        rb = tk.Radiobutton(root,
                            text=label,
                            variable=selection,
                            command=show_choice,
                            value=value,
                            **item_params)
        rb.pack(anchor=tk.W)

    button = tk.Button(root,
                       text=button_text,
                       command=finish,
                       padx=indent_px, pady=5)
    button.pack(pady=20)

    root.mainloop()
    return final_choice[0]


def _test_picker():  # requires user interaction
    for _ in range(3):
        choice = choose_item_dialog("Please select one\nof the following:",
                                    label_params=dict(justify=tk.LEFT),
                                    labels=['item 1', 'item 2 is longer', 'item 3'],
                                    values=[1, 2, 3])
        print("User chose %i." % (choice,))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_picker()
