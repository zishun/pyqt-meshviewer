"""
===========
SMPL VIEWER
===========

Zishun Liu <liuzishun@gmail.com>
Date: Feb 25, 2021
"""

import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtCore, QtGui
import numpy as np
import openmesh as om
from pyrr import Matrix44
from scipy.spatial.transform import Rotation as R

from ArcBall import ArcBallUtil
from smpl_np import SMPLModel


class StaticSettings:

    def __init__(self):
        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.joint_names = [
            '0 Pelvis',
            '1 Left Hip',
            '2 Right Hip',
            '3 Spine1',
            '4 Left Knee',
            '5 Right Knee',
            '6 Spine2',
            '7 Left Ankle',
            '8 Right Ankle',
            '9 Spine3',
            '10 Left Foot',
            '11 Right Foot',
            '12 Neck',
            '13 Left Collar',
            '14 Right Collar',
            '15 Head',
            '16 Left Shoulder',
            '17 Right Shoulder',
            '18 Left Elbow',
            '19 Right Elbow',
            '20 Left Wrist',
            '21 Right Wrist',
            '22 Left Hand',
            '23 Right Hand',
            ]
        self.initial_joint = 3


class QGLControllerWidget(QtOpenGL.QGLWidget):

    def __init__(self, parent=None):
        self.parent = parent
        super(QGLControllerWidget, self).__init__(parent)
        self.smpl = None
        self.mesh = None
        self.ssettings = StaticSettings()
        self.beta = np.zeros(tuple(self.ssettings.beta_shape))
        self.pose = np.zeros(tuple(self.ssettings.pose_shape))
        
    def initializeGL(self):
        self.ctx = moderngl.create_context()

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                uniform vec3 Light;
                in vec3 in_position;
                in vec3 in_normal;
                out vec3 v_vert;
                out vec3 v_norm;
                void main() {
                    v_norm = mat3(Mvp) * in_normal;
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = vec3(gl_Position);
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec4 Color;
                uniform vec3 Light;
                in vec3 v_vert;
                in vec3 v_norm;
                out vec4 f_color;
                void main() {
                    float lum = dot(normalize(v_norm),
                                    normalize(v_vert - Light));
                    lum = acos(lum) / 3.14159265;
                    lum = clamp(lum, 0.0, 1.0);
                    lum = lum * lum;
                    lum = smoothstep(0.0, 1.0, lum);
                    lum *= smoothstep(0.0, 80.0, v_vert.z) * 0.3 + 0.7;
                    lum = lum * 0.8 + 0.2;
                    vec3 color = Color.rgb * Color.a;
                    f_color = vec4(color * lum, 1.0);
                }
            '''
        )

        self.light = self.prog['Light']
        self.color = self.prog['Color']
        self.mvp = self.prog['Mvp']
        self.arc_ball = ArcBallUtil(2, 2)  # self.width(), self.height())
        self.set_mesh()
        self.set_joint_drawing()
        self.lookat = Matrix44.look_at(
            (0.0, 0.0, 3.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

    def set_mesh(self):
        if self.smpl is None:
            return
        self.mesh = om.TriMesh(self.smpl.verts, self.smpl.faces)
        self.mesh.update_normals()
        index_buffer = self.ctx.buffer(
            np.array(self.mesh.face_vertex_indices(), dtype="u4").tobytes())
        vao_content = [
            (self.ctx.buffer(
                np.array(self.mesh.points(), dtype="f4").tobytes()),
                '3f', 'in_position'),
            (self.ctx.buffer(
                np.array(self.mesh.vertex_normals(), dtype="f4").tobytes()),
                '3f', 'in_normal')
        ]
        self.vao = self.ctx.vertex_array(
                self.prog, vao_content, index_buffer, 4,
            )

    def set_joint_drawing(self):
        self.prog_j = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_color;
                out vec3 v_color;
                void main() {
                    v_color = in_color;
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330
                in vec3 v_color;
                out vec4 color;
                void main() {
                    color = vec4(v_color, 1.0);
                }
            '''
        )

        self.mvp_j = self.prog_j['Mvp']
        self.arc_ball_j = ArcBallUtil(2, 2)
        self.set_active_joint(self.ssettings.initial_joint)
        frame = np.zeros((6, 3), 'f4')
        frame[1, 0] = 1.0
        frame[3, 1] = 1.0
        frame[5, 2] = 1.0
        frame *= 0.6
        color = np.zeros((6, 3), 'f4')
        color[:2, 0] = 1.0
        color[2:4, 1] = 1.0
        color[4:, 2] = 1.0

        vao_content = [
            (self.ctx.buffer(frame.tobytes()), '3f', 'in_position'),
            (self.ctx.buffer(color.tobytes()), '3f', 'in_color')]
        self.vao_j = self.ctx.vertex_array(self.prog_j, vao_content)

    def paintGL(self):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        if self.mesh is None:
            return

        self.aspect_ratio = self.width()/max(1.0, self.height())
        self.proj = Matrix44.perspective_projection(60.0, self.aspect_ratio,
                                                    0.1, 1000.0)

        self.light.value = (0.5, 1.0, 1.0)
        self.color.value = (1.0, 1.0, 1.0, 0.8)
        mvp = self.proj * self.lookat * self.arc_ball.Transform
        self.mvp.write(mvp.astype('f4'))

        self.vao.render()
        self.draw_joint()
        self.ctx.finish()

    def draw_joint(self):
        mvp = self.proj * self.lookat * self.arc_ball.Transform * \
            self.arc_ball_j.Transform
        self.mvp_j.write(mvp.astype('f4'))
        self.vao_j.render(moderngl.LINES, 3 * 2)
        # use geometry shader to change the line width

    def resizeGL(self, Width, Height):
        Height = max(2, Height)
        Width = max(2, Width)
        self.ctx.viewport = (0, 0, Width, Height)
        self.arc_ball.setBounds(Width, Height)
        return

    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftDown(event.x(), event.y())
        if event.buttons() & QtCore.Qt.RightButton:
            self.arc_ball_j.onClickLeftDown(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftUp()
        if event.buttons() & QtCore.Qt.RightButton:
            self.arc_ball_j.onClickLeftUp()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
                self.arc_ball.onDrag(event.x(), event.y())
        if event.buttons() & QtCore.Qt.RightButton:
            self.arc_ball_j.onDrag(event.x(), event.y())
            r = R.from_matrix(
                self.arc_ball_j.Transform[:3, :3].T).as_rotvec()
            self.pose[self.active_joint*3:self.active_joint*3+3] = r
            self.update_mesh_data()

    def set_beta(self, val, i):
        scale = 3
        self.beta[i] = (val/100.0-0.5)*2*scale
        self.update_mesh_data()

    def update_mesh_data(self):
        if self.mesh is None:
            return
        self.smpl.set_params(beta=self.beta, pose=self.pose)
        self.set_mesh()
        self.arc_ball_j.Transform = self.smpl.G[self.active_joint].T

    def set_active_joint(self, i):
        self.active_joint = i
        if self.smpl is not None:
            self.arc_ball_j.ThisRot = self.smpl.R[self.active_joint, :3, :3].T
            self.arc_ball_j.Transform = self.smpl.G[self.active_joint].T
            self.arc_ball_j.setBounds(self.width(), self.height())

    def load_SMPL(self, fn):
        self.smpl = SMPLModel(fn)
        self.beta = np.zeros((self.smpl.beta.size))
        self.pose = np.zeros((self.smpl.pose.size))
        self.set_mesh()
        self.set_active_joint(self.active_joint)

    def load_params(self, fn):
        data = np.load(fn)
        self.beta = data['beta']
        self.pose = data['pose']
        self.update_mesh_data()
        self.set_active_joint(self.active_joint)

    def write_params(self, fn):
        np.savez(fn, beta=self.beta, pose=self.pose)

    def write_mesh(self, fn):
        om.write_mesh(fn, self.mesh)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('SMPL Viewer')
        self.setWindowIcon(QtGui.QIcon('icon.png'))

        self.gl = QGLControllerWidget(self)

        self.initGUI()
        self.initMenu()

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.gl.updateGL)
        timer.start()

    def initGUI(self):
        lvbox = QtWidgets.QVBoxLayout()

        # left vbox part 1: shape sliders
        shapeGroupBox = QtWidgets.QGroupBox("Shape Settings")
        groupVbox = QtWidgets.QVBoxLayout()
        sliders = []
        for i in range(self.gl.ssettings.beta_shape[0]):
            sliders.append(QtWidgets.QSlider(QtCore.Qt.Horizontal))
        for i in range(len(sliders)):
            sliders[i].setValue(50)
            sliders[i].valueChanged.connect(
                lambda val, x=i: self.gl.set_beta(val, x))
            groupVbox.addWidget(sliders[i])
        shapeGroupBox.setLayout(groupVbox)
        lvbox.addWidget(shapeGroupBox)
        
        # left vbox part 2: pose
        poseGroupBox = QtWidgets.QGroupBox("Pose Settings")
        groupVbox = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel()
        label.setText('Current Joint')
        groupVbox.addWidget(label)
        cb = QtWidgets.QComboBox()
        cb.addItems(self.gl.ssettings.joint_names)
        cb.setCurrentIndex(self.gl.ssettings.initial_joint)
        cb.currentIndexChanged.connect(self.gl.set_active_joint)
        groupVbox.addWidget(cb)
        poseGroupBox.setLayout(groupVbox)
        lvbox.addWidget(poseGroupBox)

        lvbox.addStretch(1)

        chbox = QtWidgets.QHBoxLayout()
        chbox.addLayout(lvbox, 1)
        chbox.addWidget(self.gl, 3)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(chbox)

        self.setCentralWidget(central_widget)

    def initMenu(self):
        self.menu = self.menuBar().addMenu("&File")
        self.menu.addAction('&Open SMPL', self.openSMPL)
        self.menu.addAction('&Load Params', self.loadParams)
        self.menu.addAction('&Save Params', self.saveParams)
        self.menu.addAction('&Save Mesh', self.saveMesh)

    def openSMPL(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open SMPL', '', "SMPL files (*.pkl)")
        self.gl.load_SMPL(fname[0])

    def loadParams(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open body parameters', '', "NPZ files (*.npz)")
        self.gl.load_params(fname[0])

    def saveParams(self):
        fname = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Write body parameters', '', "NPZ files (*.npz)")
        self.gl.write_params(fname[0])

    def saveMesh(self):
        fname = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Write mesh', '', "Mesh files (*.obj *.off *.stl *.ply)")
        self.gl.write_mesh(fname[0])


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    app.exec()
