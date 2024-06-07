# Common gui utilities

## Camera

`camera.py` - Wrapper for OpenCV `VideoCapture()`, providing management of the settings. Creating a 
`Camera()` object will look for this information in two configuration files, or scan it from the os and write it to 
those configuration files. The two files are:

* `~/.camera.system_config.json` containing the number of cameras are available to the OS and their resolution
  capabilities.   Delete this file to re-scan your computer's camera capabilities the next time a
`Camera()` object is created.
* `~/.camera.user_config.json` indicating which camera is to be used and what resolution it should be set to.  If 
  this information is present and complete, the `Camera()` object is returned immediately, otherwise the user is asked 
(via UI window or console) to provide it first.  Delete this file to select a new camera configuration the next time a
`Camera()` object is created.

** NOTE:  Scanning cameras appears to be much slower in Windows 11 than in other operating
systems. To speed things up, running w/this OS will scan an abbreviated list of resolutions
to test.  (To use the full list anyway, overwrite that file, `common_resolutions_abbrev.json`, with the contents of the
full resolution file `common_resolutions.json`.

### Demo
From path 1 level above `/gui_utils/`:

```
 > python -m gui_utils.test_camera
```

![res_detect_user_pick](https://github.com/andsmith/gui_utils/blob/main/detect_resolution_and_gui_picker.png)

## menu components

* `gui_picker.py` - Dialog box for choosing an item from a list (of text or images / icons)

* `text_annotation.py` - Display messages with optional lifetimes in transparent boxes.

## Coordinate grids control/display

* Collect/show data from a cartesian grid (only implemented one).
* Keyboard control of axis resizing

* `coordinate_grid.py`:  Control a 2-d value in real time on a scalable, user-defined grid.

### Demo
  Run `python -m gui_utils.coordinate_grids.py` for this demo:
  ![demo in coordinate_grids.py](https://github.com/andsmith/gui_utils/blob/main/grid.png)
