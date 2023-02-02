# Common gui utilities

## Camera
* `camera.py` - Manager for OpenCV `VideoCapture()`
  * Count cameras attached.
  * Probe for resolutions & ask user to pick one, etc.
  * Live switching between cameras
  * Change settings such as resolution & FPS
  
Detect resolutions & use the gui_picker to select one:

![res_detect_user_pick](https://github.com/andsmith/gui_utils/blob/main/detect_resolution_and_gui_picker.png)

Run as module for this demo:  `python -m gui_utils.camera_settings`
    
## menu components
* `gui_picker.py` - Dialog box for choosing an item from a list
Camera component for projects
* `gui_icon_overlay.py` - Overlay icon(s) with interativity on a video stream
* `text_annotation.py` - Display messages with optional lifetimes in transparent boxes.
## Coordinate grids control/display
* Collect/show data from a cartesian grid (only implemented one).
* Keyboard control of axis resizing

* `coordinate_grid.py`:  Control a 2-d value in real time on a scalable, user-defined grid. 
Run `python -m gui_utils.coordinate_grids.py` for this demo:
![demo in coordinate_grids.py](https://github.com/andsmith/gui_utils/blob/main/grid.png)
