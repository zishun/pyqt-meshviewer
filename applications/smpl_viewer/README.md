# SMPL Viewer

<img src="https://github.com/zishun/pyqt-meshviewer/raw/master/imgs/screenshot_smpl.png" width="350"/>

## Dependencies

See [README](../../) in the root folder of this repository. Test by running the mesh viewer ```mgl_qt_meshviewer.py```.


## Usage

1. Prepare the SMPL model file (.pkl) following CalciferZh's [instructions](https://github.com/CalciferZh/SMPL)
2. Open the viewer by ```python ./SMPLViewer.py```.
3. Click ```File > Open SMPL``` in the menu bar to load the prepared pkl file.
4. Body manipulation
    * Body shape: use slide bars in the ```Shape Settings``` block.
    * Body pose: set ```Current Joint``` in the ```Pose Settings``` block. Drag the right mouse button to rotate the joint. Drag left mouse button to change the view.
5. File IO (in the ```File``` menu)
    * Read/write body shape and pose parameters from/to an NPZ file.
    * Write body geometry to a mesh file.
