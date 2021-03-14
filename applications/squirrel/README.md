# Spatial Keyframing for Performance-driven Animation

This is an implementation of the animation method presented in

> Takeo Igarashi, Tomer Moscovich, John F. Hughes, "Spatial Keyframing for Performance-driven Animation", ACM SIGGRAPH / Eurographics Symposium on Computer Animation, 2005.

<img src="https://github.com/zishun/pyqt-meshviewer/raw/master/imgs/screenshot_squirrel.gif" width="280"/>

## Usage

1. Dependencies: see [README](../../) in the root folder of this repository. Test by running the mesh viewer ```mgl_qt_meshviewer.py```.
2. Data: download the demonstration software from the [official project page](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/research/squirrel/index.html). Unzip it here and get.

    ```
    pyqt-meshviewer/applications/spatial_keyframing/
     |
     ├──keyframes.py
     ├──SquirrelViewer.py
     ├──...
     ├──...
     └──squirrel/
         ├──data/
         ├──lib/
         ├──manual/
         ├──...
         └──...
    ```

3. Run ```python ./SquirrelViewer.py```.
    * Click ```File > Open``` to load a ```.key``` file.
    * Drag the right mouse button to move the control cursor
    * Drag the left mouse button to change the view.


## Notes

* Only for viewing. Cannot create new keyframes.
* Only geometry data in the ```.asg``` files is loaded. Color/texture information is skipped.
* To handle the degenerated cases described at the end of section 4.1, small singular values in the linear system are cut-off, which is different from the original paper.
* SVD decomposition is used for orthonormalization, different from the method in section 4.2.


## Links

* [Project page by the original authors](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/research/squirrel/index.html)
* [Video of another implementation](https://youtu.be/eJ9Pt6R-ank)
