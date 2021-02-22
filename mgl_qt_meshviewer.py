import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtCore
import numpy as np
import openmesh as om
from pyrr import Matrix44

from ArcBall import ArcBallUtil


class QGLControllerWidget(QtOpenGL.QGLWidget):

    def __init__(self, parent=None):
        self.parent = parent
        super(QGLControllerWidget, self).__init__(parent)

    def initializeGL(self):
        self.ctx = moderngl.create_context()

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;
                out vec3 v_vert;
                out vec3 v_norm;
                void main() {
                    v_vert = in_position;
                    v_norm = in_normal;
                    gl_Position = Mvp * vec4(in_position, 1.0);
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
        self.mesh = None
        self.arc_ball = ArcBallUtil(self.width(), self.height())
        self.center = np.zeros(3)
        self.scale = 1.0

    def set_mesh(self, mesh):
        self.mesh = mesh
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
        self.init_arcball()

    def init_arcball(self):
        self.arc_ball = ArcBallUtil(self.width(), self.height())
        pts = self.mesh.points()
        bbmin = np.min(pts, axis=0)
        bbmax = np.max(pts, axis=0)
        self.center = 0.5*(bbmax+bbmin)
        self.scale = np.linalg.norm(bbmax-self.center)
        self.arc_ball.Transform[:3, :3] /= self.scale
        self.arc_ball.Transform[3, :3] = -self.center/self.scale

    def paintGL(self):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)

        if self.mesh is None:
            return

        self.aspect_ratio = self.width()/max(1.0, self.height())
        proj = Matrix44.perspective_projection(60.0, self.aspect_ratio,
                                               0.1, 1000.0)
        lookat = Matrix44.look_at(
            (0.0, 0.0, 2.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        self.light.value = (1.0, 1.0, 1.0)
        self.color.value = (1.0, 1.0, 1.0, 0.8)
        self.arc_ball.Transform[3, :3] = \
            -self.arc_ball.Transform[:3, :3].T@self.center
        self.mvp.write(
            (proj * lookat * self.arc_ball.Transform).astype('f4'))

        self.vao.render()

    def resizeGL(self, Width, Height):
        if Height == 0:
            Height = 1
        self.ctx.viewport = (0, 0, Width, Height)
        self.arc_ball.setBounds(Width, Height)
        return

    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            # pos = event.pos()
            self.arc_ball.onclickLeftDown(event.x(), event.y())
            # self.clicked.emit()

    def mouseReleaseEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftUp()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onDrag(event.x(), event.y())


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Mesh Viewer')
        self.gl = QGLControllerWidget(self)

        self.setCentralWidget(self.gl)
        self.menu = self.menuBar().addMenu("&File")
        self.menu.addAction('&Open', self.openFile)

        timer = QtCore.QTimer(self)
        timer.setInterval(20)  # period, in milliseconds
        timer.timeout.connect(self.gl.updateGL)
        timer.start()

    def openFile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', '', "Mesh files (*.obj *.off *.stl *.ply)")
        mesh = om.read_trimesh(fname[0])
        self.gl.set_mesh(mesh)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    win = MainWindow()

    win.show()
    app.exec()
