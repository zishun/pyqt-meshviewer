import numpy as np
import openmesh as om
# import networkx as nx
from zipfile import ZipFile
import tempfile
from xml.etree import cElementTree as ElementTree


class ASGLoader:

    def __init__(self,):
        pass

    def get_class(self, c):
        return c.split('.')[-1]
    
    def get_Matrix4d(self, e):
        m = np.empty((4, 4))
        cnt = 0
        for r in e:
            m[cnt] = np.fromstring(r.text, sep=' ')
            cnt += 1
        return m

    def get_verterices(self, e):
        verts = []
        verts_n = []
        for x in e:
            if x.tag == 'vertex':
                for y in x:
                    if y.tag == 'position':
                        verts.append(np.fromstring(y.text, sep=' '))
                    elif y.tag == 'normal':
                        verts_n.append(np.fromstring(y.text, sep=' '))
        return np.array(verts), np.array(verts_n)

    def get_IndexedTriangleArray(self, e):
        for x in e:
            if 'name' in x.attrib:
                name = x.attrib['name']
                #if name == 'Indices':
                if name == 'Vertices':
                    v, vn = self.get_verterices(x)
        self.verts[self.get_key(e)] = v
        self.verts_n[self.get_key(e)] = vn

    def get_VisualGeometry(self, e):
        for x in e:
            if 'name' in x.attrib:
                if x.attrib['name'] == 'Geometry':
                    return self.get_key(x)

    def get_key(self, e):
        return int(e.attrib['key'])

    def load_asg(self, fn):
        # print('Warning! Vertex colors and other attributes in asg file are not loaded.')
        # .asg is a zipped xml file
        with tempfile.TemporaryDirectory() as tmpdir:
            with ZipFile(fn, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
                self.load_xml(tmpdir + '/root.xml')
        return self.verts, self.vn, self.faces, self.part_sizes, \
            self.parent, self.transforms, self.order

    def load_xml(self, fn):
        tree = ElementTree.parse(fn)
        root = tree.getroot()
        self.verts = {}
        self.verts_n = {}
        self.Transformable2Geometry = {}
        self.TransformableParent = {}
        self.Transforms = {}
        self.order = []
        self.read_node(root)
        self.collect()

    def read_node(self, root, parent=-1):
        # print(self.get_class(root.attrib['class']))
        key = self.get_key(root)
        self.order.append(key)
        self.TransformableParent[key] = parent
        for x in root:
            # print(x.tag, x.attrib)
            if 'class' in x.attrib:
                c = self.get_class(x.attrib['class'])
                # print(c)
                if c == 'Transformable':
                    self.read_node(x, key)
                elif c == 'Matrix4d':
                    self.Transforms[key] = self.get_Matrix4d(x)
                elif c == 'IndexedTriangleArray':
                    self.get_IndexedTriangleArray(x)
                elif c == 'Visual':
                    g = self.get_VisualGeometry(x)
                    self.Transformable2Geometry[key] = g

    def collect(self):
        #DG = nx.DiGraph()
        #for x in self.TransformableParent:
        #    if self.TransformableParent[x] > 0:
        #        DG.add_edge(self.TransformableParent[x], x)
        #order = list(nx.topological_sort(DG))
        order = self.order
        cnt = [0]
        for x in order:
            cnt.append(cnt[-1] + self.verts[self.Transformable2Geometry[x]].shape[0])
        self.part_sizes = np.array(cnt, 'i4')
        self.verts = np.vstack(tuple([self.verts[self.Transformable2Geometry[x]] for x in order]))
        self.vn = np.vstack(tuple([self.verts_n[self.Transformable2Geometry[x]] for x in order]))
        self.faces = np.vstack(tuple(
            [np.arange(self.part_sizes[i], self.part_sizes[i+1]).reshape((-1, 3)) for i in range(self.part_sizes.size-1)]
        )).astype('i4')
        # mesh = om.TriMesh(self.verts, self.faces)
        # mesh.update_normals()
        # self.vn = mesh.vertex_normals()
        # om.write_mesh('test.obj', mesh)

        self.transforms = np.concatenate(tuple([self.Transforms[x].reshape((1, 4, 4)) for x in order]))
        key2idx = {}
        for i in range(len(order)):
            key2idx[order[i]] = i
        self.parent = np.full((self.part_sizes.size-1,), -1, dtype='i4')
        for c in order[1:]:
            p = self.TransformableParent[c]
            self.parent[key2idx[c]] = key2idx[p]
        self.order = [i for i in range(self.part_sizes.size-1)]


if __name__ == '__main__':
    loader = ASGLoader()
    loader.load_xml('squirrel/data/bear/root.xml')
