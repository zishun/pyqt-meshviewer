"""
==================
SAPTIAL kEYFRAMING
==================

based on the method presented in:
Takeo Igarashi, Tomer Moscovich, John F. Hughes, "Spatial Keyframing for
Performance-driven Animation", SCA 2005.

Zishun Liu <liuzishun@gmail.com>
Date: Feb 28, 2021
"""


import numpy as np
import openmesh as om
from scipy.spatial.distance import cdist


class Keyframes:

    def __init__(self):
        pass

    def load_obj(self, fn):
        vert = []
        face = []
        cnt = []
        transforms = []
        children = []
        with open(fn, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            s = lines[i].split()
            if len(s) == 0:
                continue
            if s[0] == 'v':
                vert.append([float(x) for x in s[1:4]])
            elif s[0] == 'f':
                f_list = list(map(lambda x: int(x.split('/')[0]),  s[1:]))
                face.append(f_list)
            elif s[0] == '#':
                if s[1] == 'id':
                    if int(s[2]) == len(cnt):
                        cnt.append(len(vert))
                    else:
                        print('Warning! Inconsistent part index')
                elif s[1] == 'children':
                    children.append([int(x) for x in s[2:]])
                elif s[1] == 'transformable':
                    mat = np.fromstring(lines[i][16:], sep=' ').reshape((4,4))
                    transforms.append(mat)  # [R t].T
        cnt.append(len(vert))

        self.verts = np.array(vert)
        self.faces = np.array(face, 'i4')-1  # 1-indexed -> 0-indexed
        self.part_sizes = np.array(cnt, 'i4')
        self.transforms = np.array(transforms)
        self.parent = np.full((self.part_sizes.size-1,), -1, dtype='i4')
        edges = []
        need_toposort = False
        for i in range(len(children)):
            for j in children[i]:
                self.parent[j] = i
                edges.append([i,j])
                if j < i:
                    need_toposort = True
        if need_toposort:
            print('Need topological_sort')
            import networkx as nx
            DG = nx.DiGraph(edges)
            self.order = list(nx.topological_sort(DG))
        else:
            self.order = [x for x in range(self.part_sizes.size-1)]

    def load_keys(self, fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        
        # header
        s = lines[0].split()
        assert(s[0] == 'balls')
        self.num_keys = int(s[1])
        s = lines[1].split()
        assert(s[0] == 'models')
        self.num_parts = int(s[1])

        # content
        self.balls = np.empty((self.num_keys, 3))
        self.poses = np.empty((self.num_keys, self.num_parts, 4, 4))
        cnt = 2
        for i in range(self.num_keys):
            self.balls[i, :] = np.fromstring(lines[cnt], sep=' ')
            for j in range(self.num_parts):
                mat = np.fromstring(lines[cnt+1+j], sep=' ')
                self.poses[i, j, :, :] = mat.reshape((4,4))  # [R t].T
            cnt += 1+self.num_parts

    def load_asg(self, fn):
        print('asg file loading not implemented!', fn)
        # .asg is a zipped xml file

    def update_mesh(self, pose=None):
        if pose is None:
            pose = self.transforms
        v = np.empty_like(self.verts)
        G = np.empty((self.part_sizes.size-1, 4, 4))
        G[0] = pose[self.order[0]]
        for i in range(1, len(self.order)):
            G[i] = pose[self.order[i]] @ G[self.parent[self.order[i]]]

        for i in range(len(self.order)):
            v[self.part_sizes[i]:self.part_sizes[i+1]] = \
                self.verts[self.part_sizes[i]:self.part_sizes[i+1]] @ G[i, :3, :3] \
                + G[i, 3, :3]
        # mesh = om.TriMesh(v, self.faces)
        # om.write_mesh('../data/test.obj', mesh)
        return v

    def prepare_interpolation(self):
        # TODO: sec4.1 Special care must be taken when there are fewer than
        # four spatial keyframes and when the spatial distribution of the
        # markers is degenerate 

        n = self.num_keys
        A = np.zeros((n+4, n+4))
        A[:n, :n] = cdist(self.balls, self.balls)
        A[n, :n] = 1
        A[:n, n] = 1
        A[:n, n+1:n+4] = self.balls
        A[n+1:n+4, :n] = self.balls.T

        B = np.zeros((n+4, 12*self.num_parts))
        B[:n, :] = self.poses[:, :, :, :3].reshape((n, -1))
        #self.coeff = np.linalg.solve(A, B)
        self.coeff = np.linalg.lstsq(A, B, rcond=None)[0]

    def interpolate(self, p):
        n = self.num_keys
        A = np.empty((1, n+4))
        A[0, :n] = np.linalg.norm(self.balls-p, axis=1)
        A[0, n] = 1
        A[0, n+1:n+4] = p
        pose = A @ self.coeff
        pose = pose.reshape((self.num_parts, 4, 3))
        pose = np.concatenate((pose, self.transforms[:, :, -1:]), axis=2)

        # Orthonormalization, different from the paper
        for i in range(self.num_parts):
            u, _, vh = np.linalg.svd(pose[i, :3, :3])
            pose[i, :3, :3] = u @ vh

        return self.update_mesh(pose)


if __name__ == '__main__':
    keyframes = Keyframes()
    keyframes.load_obj('./squirrel/data/squirrel.obj')
    # print(keyframes.verts.shape, keyframes.faces.shape, keyframes.part_sizes)
    keyframes.load_keys('./squirrel/data/squirrel_run.key')
    # print(keyframes.balls, keyframes.poses)
    # keyframes.update_mesh(keyframes.poses[2])
    keyframes.update_mesh(keyframes.transforms)
