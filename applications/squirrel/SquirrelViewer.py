import moderngl
from PyQt5 import QtOpenGL, QtWidgets, QtCore
import numpy as np
import openmesh as om
from pyrr import Matrix44
import os

from ArcBall import ArcBallUtil
from keyframes import Keyframes


class QGLControllerWidget(QtOpenGL.QGLWidget):

    def __init__(self, parent=None):
        # TODO: parent?
        #self.parent = parent
        #super(QGLControllerWidget, self).__init__(parent)
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setDoubleBuffer(True)
        super(QGLControllerWidget, self).__init__(fmt, None)
        self.keyframes = Keyframes()
        self.balls = None

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

        # https://www.labri.fr/perso/nrougier/python-opengl/#spheres
        self.prog_balls = self.ctx.program(
            vertex_shader='''
                #version 330
                uniform mat4 Mvp;
                in vec3 in_vert;
                in vec3 in_color;
                out vec3 v_center;
                out vec3 v_color;
                void main() {
                    gl_PointSize = 20;
                    gl_Position = Mvp * vec4(in_vert, 1.0);
                    v_center = vec3(gl_Position);
                    v_color = in_color;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec3 Light;
                // uniform float Radius;
                in vec3 v_center;
                in vec3 v_color;
                out vec4 frag_color;
                void main()
                {
                    vec2 p = gl_PointCoord.xy - vec2(0.5);
                    float z = 0.5 - length(p);
                    if (z < 0) discard;

                    // gl_FragDepth is ignored
                    vec3 normal = normalize(vec3(p.x, -p.y, z)); // https://github.com/moderngl/moderngl/issues/253#issuecomment-577293096
                    vec3 direction = normalize(Light);
                    float diffuse = max(0.0, dot(direction, normal));
                    float specular = pow(diffuse, 24.0);
                    frag_color = vec4(max(diffuse*v_color, specular*vec3(1.0)), 1.0);
                }
            '''
        )

        self.mesh = None
        self.arc_ball = ArcBallUtil(self.width(), self.height())
        self.center = np.zeros(3)
        self.scale = 1.0

    def read_key(self, fn):
        self.keyframes.load_keys(fn)

        path, fn = os.path.split(fn)
        name, _ = os.path.splitext(fn)
        obj_fn = os.path.join(path, name.split('_')[0]+'.obj')
        if os.path.isfile(obj_fn):
            self.keyframes.load_obj(obj_fn)
        else:
            self.keyframes.load_asg(obj_fn[:-3]+'asg')
        v = self.keyframes.update_mesh()
        mesh = om.TriMesh(v, self.keyframes.faces)
        balls = self.keyframes.balls
        self.set_mesh(mesh, balls)
        self.keyframes.prepare_interpolation()

    def set_mesh(self, mesh, balls):
        self.mesh = mesh
        self.mesh.update_normals()
        assert(self.mesh.n_vertices() > 0 and self.mesh.n_faces() > 0)
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

        self.balls = np.vstack((balls, balls[-1:])).astype('f4')  # the last for moving
        self.balls_color = np.zeros_like(self.balls, 'f4')
        self.balls_color[:, 0] = 1.0
        self.balls_color[:-1, 1] = 1.0
        vao_content_ball = [
            (self.ctx.buffer(self.balls.tobytes()), '3f', 'in_vert'),
            (self.ctx.buffer(self.balls_color.tobytes()), '3f', 'in_color')]
        self.vao_balls = self.ctx.vertex_array(
                self.prog_balls, vao_content_ball
            )
        self.init_arcball()

    def init_arcball(self):
        self.arc_ball = ArcBallUtil(self.width(), self.height())
        pts = self.mesh.points()
        bbmin = np.min(pts, axis=0)
        bbmax = np.max(pts, axis=0)
        self.center = 0.5*(bbmax+bbmin)
        self.scale = np.linalg.norm(bbmax-self.center)
        lookat = Matrix44.look_at(
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        self.arc_ball.ThisRot[:, :] = lookat[:3, :3]
        self.arc_ball.Transform[:3, :3] = self.arc_ball.ThisRot/self.scale
        self.arc_ball.Transform[3, :3] = -self.center/self.scale

    def paintGL(self):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | 
            moderngl.BLEND)

        if self.mesh is None:
            return

        self.aspect_ratio = self.width()/max(1.0, self.height())
        proj = Matrix44.perspective_projection(60.0, self.aspect_ratio,
                                               0.1, 1000.0)
        lookat = Matrix44.look_at(
            (0.0, 0.0, 3.0),
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )

        self.light.value = (1.0, 1.0, 1.0)
        self.color.value = (1.0, 1.0, 1.0, 0.8)
        self.prog_balls['Light'].value = self.light.value
        #self.prog_balls['Radius'].value = 100.0
        self.arc_ball.Transform[3, :3] = \
            -self.arc_ball.Transform[:3, :3].T@self.center
        self.mvp_mat = (proj * lookat * self.arc_ball.Transform).astype('f4')
        self.mvp.write(self.mvp_mat)
        self.prog_balls['Mvp'].write(self.mvp_mat)

        vao_content_ball = [
            (self.ctx.buffer(self.balls.tobytes()), '3f', 'in_vert'),
            (self.ctx.buffer(self.balls_color.tobytes()), '3f', 'in_color')]
        self.vao_balls = self.ctx.vertex_array(
                self.prog_balls, vao_content_ball
            )

        self.vao.render()
        self.vao_balls.render(moderngl.POINTS, self.balls.shape[0])
        self.ctx.finish()

    def resizeGL(self, width, height):
        width = max(2, width)
        height = max(2, height)
        self.ctx.viewport = (0, 0, width, height)
        self.arc_ball.setBounds(width, height)
        return

    def pickLocation(self, x, y):
        if self.balls is None:
            return

        c = self.balls[:-1].mean(axis=0)
        c = self.mvp_mat.T @ np.hstack((c, 1.0))
        z = c[2] / c[3]

        x = x/self.width()*2-1
        y = -y/self.height()*2+1
        #X = np.linalg.lstsq(self.mvp_mat.T, np.array([x, y, z, 1.0]), rcond=None)[0]
        X = np.linalg.solve(self.mvp_mat.T, np.array([x, y, z, 1.0]))
        p = X[:3] / X[3]
        self.balls[-1] = p
        v = self.keyframes.interpolate(p)
        
        self.mesh = om.TriMesh(v, self.keyframes.faces)
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


    def mousePressEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftDown(event.x(), event.y())
        elif event.buttons() & QtCore.Qt.RightButton:
            self.pickLocation(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onClickLeftUp()

    def mouseMoveEvent(self, event):
        if event.buttons() & QtCore.Qt.LeftButton:
            self.arc_ball.onDrag(event.x(), event.y())
        elif event.buttons() & QtCore.Qt.RightButton:
            self.pickLocation(event.x(), event.y())


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.resize(640, 480)
        self.setWindowTitle('Squirrel Viewer')
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
            self, 'Open file', '', "key files (*.key)")
        self.gl.read_key(fname[0])


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = MainWindow()

    win.show()
    app.exec()
