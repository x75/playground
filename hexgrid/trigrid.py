"""triangular grid

see trimesh, networkx
 - load mesh
 - get triangle graph with neighbors
 - create the mesh as a graph
 - compute cells, render cell state onto mesh

"""

from __future__ import division
from __future__ import absolute_import

import pickle, sys, time, threading, argparse

import numpy as np
import numpy.linalg as la
from six.moves import range
from pprint import pformat

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from scipy.spatial.transform import Rotation as R
import joblib
import matplotlib.pyplot as plt

# use meshpy
import meshpy.triangle as triangle
from meshpy.triangle import MeshInfo, build
# use trimesh
import trimesh

# meshpy examples toolkit
import jw_meshtools as mt

def dl2ld(DL):
    v = [dict(zip(DL,t)) for t in zip(*DL.values())]
    return v

def ld2dl(LD):
    v = {k: [dic[k] for dic in LD] for k in LD[0]}
    return v
    
def make_vertex_facets_hexagon(params):
    """make_vertex_facets_hexagon

    create a list of 2D vertices and line facets that make up a hexagon
    """
    # vertices, points
    dim = params['dim']
    if dim == 2:
        points = [(0, 0)]
        for ang in [0, 60, 120, 180, 240, 300]:
            points.append(
                (np.cos(np.deg2rad(ang)) * params['c'],
                 np.sin(np.deg2rad(ang)) * params['c']
                )
            )
    else:
        points = [(0, 0, 0)]
        for ang in [0, 60, 120, 180, 240, 300]:
            points.append(
                (np.cos(np.deg2rad(ang)) * params['c'],
                 np.sin(np.deg2rad(ang)) * params['c'],
                 0.0)
            )
            
    # print('points = {0}'.format(pformat(points)))
    # facets := set of point pairs defining what again?
    facets = [
        [0,1], [0,2], [0,3],
        [0,4], [0,5], [0,6],
        [1,2], [2,3], [3,4],
        [4,5], [5,6], [6,1],
        # [2,3],
        # [3,4],
        # [4,5],
        # [5,6],
        # [6,4],
    ]
    # set of edges
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 5],
        [0, 5, 6],
        [0, 1, 6],
    ]
    
    return (points, facets, faces)

def make_vertex_facets_line(params):
    """make_vertex_facets_line

    create a list of 2d points and triangle facets that make up a line built
    """
    dim = params['dim']
    if dim == 2:
        points = [(0, 0)]
        for trans in [0, 1, 2, 3, 4]:
            angs = [0, 60]
            # if trans > 0:
            #     angs += [120]
            
            for ang in angs:
                points.append(
                    (np.cos(np.deg2rad(ang)) * params['c'] + trans,
                     np.sin(np.deg2rad(ang)) * params['c']
                    )
                )
    else:
        points = [(0, 0, 0)]
        for trans in [0, 1, 2, 3, 4]:
            angs = [0, 60]
            # if trans > 0:
            #     angs += [120]
            
            for ang in angs:
                points.append(
                    (np.cos(np.deg2rad(ang)) * params['c'] + trans,
                     np.sin(np.deg2rad(ang)) * params['c'],
                     0.0
                    )
                )

    facets = [
        [0, 1], [0, 2], [1,2],
        [1, 2], [1, 4], [2,4],
        [1, 3], [1, 4], [3,4],
        [3, 4], [3, 6], [4,6],
        [3, 5], [3, 6], [5,6],
        [5, 6], [5, 8], [6,8],
        [5, 7], [5, 8], [7,8],
    ]

    faces = [
        [0, 1, 2],
        [1, 2, 4],
        [1, 3, 4],
        [3, 4, 6],
        [3, 5, 6],
        [5, 6, 8],
        [5, 7, 8]
    ]
    
    return (points, facets, faces)

def make_vertex_facets_rect(params, **kwargs):
    """make_vertex_facets_rect

    create a list of 2d point and triangle facets that fill up an outer rectangle
    """
    length = 0.15
    # Simple mesh rectangle
    p,v=mt.RectangleSegments([-2, -1.5],[2, 1.5], edge_length=length)
    
    # p1,v1=mt.CircleSegments([1.,0],1,a_min=-np.pi/2,a_max=np.pi/2,num_points=20)
    # p2,v2=mt.CircleSegments([1,0],3,a_min=np.pi/2.,a_max=3.*np.pi/2,num_points=20)
    # p,v=mt.AddSegments(p1,p2,closed=True)
    # p1,v1=mt.RectangleSegments([-2,-2],[2.5,3],edge_length=length)
    # p2,v2=mt.CircleSegments([1,1],1,edge_length=length/5)
    # p,v=mt.AddCurves(p1,v1,p2,v2)
    # mt.DoTriMesh(p,v,edge_length=length)
    

    # p1,v1=mt.LineSegments([-2,-3],[2,-3],num_points=12)
    # p2,v2=mt.LineSegments([2,3],[-2,3],num_points=12)
    # p,v=mt.AddSegments(p1,p2,closed=True)
    # p3,v3=mt.CircleSegments([-0.5,0.5],0.5,edge_length=length)
    # p,v=mt.AddCurves(p,v,p3,v3)
    # p4,v4=mt.CircleSegments([1,-1],0.5,edge_length=length)
    # p,v=mt.AddCurves(p,v,p4,v4)
    # mt.DoTriMesh(p,v,edge_length=length,holes=[(-0.4,0.4),(0.95,-0.8)])
    return (p, v)

def make_vertex_facets_rect_trimesh(params):
    """make_vertex_facets_rect

    create a list of 2d point and triangle facets that fill up an outer rectangle
    """
    length = 0.15
    mesh = trimesh.primitives.Box(
        center=[0, 0, 0],
        extents=[3, 3, 3],
        transform=trimesh.transformations.random_rotation_matrix(),
        sections=100,
    )
    # perim = np.random.uniform(-1, 1, (7, 3))
    # mesh = trimesh.creation.Polygon(perim)
    # mesh = trimesh.primitives.Cylinder()
    # mesh = trimesh.primitives.Capsule()
    # mesh = trimesh.primitives.Sphere()
    return mesh.vertices, None, mesh.faces

def make_vertex_facets_load(params):
    p = None
    v = None
    return (p, v)

class runUpdate(threading.Thread):
    def __init__(self, mesh, tris, density):
        super(runUpdate, self).__init__()

        self.isrunning = True
        assert mesh is not None, "Need to supply mesh argument" 
        assert tris is not None, "Need to supply tris argument" 
        self.mesh = mesh
        self.tris = tris
        self.cnt = 0
        self.density = density
        
    def run(self):
        while self.isrunning:
            # print('runUpdate')
            self.mesh_update_state(self.cnt, self.mesh, self.tris, self.density)
            self.cnt += 1
            time.sleep(1/20.)

    def mesh_update_state(self, cnt, mesh, tris, density):
        event_density = density # 0.0001
        # event_density = 0.001
        # event_density = 0.01
        # event_density = 0.1
        # for tri_i, neighbors in enumerate(mesh.neighbors):
        # print(cnt)
        for tri_i, tri in enumerate(tris):

            x_ = tris[tri_i]['state']
            # print(tri_i, x_)
            y_ = np.zeros_like(x_)
            # periodic activation

            # y_ += 0.05 * np.sin((cnt/20.0) * tris[tri_i]['freq'] * 2 * np.pi)
            # y_ += 0.3 * np.sin(cnt/1000.0)**2
            
            # if tri_i == 0 and cnt % 100 == 0:
            # if tri_i == 0 and np.random.uniform() < event_density:
            if np.random.uniform() < event_density:
                # print('refreshing state')
                # tris[tri_i]['state'] = 1.0
                y_ += 2.0 + np.random.uniform(0, 2)
                # tris[tri_i]['state'] = x_

        
            # print(tri_i, neighbors)
        
            # print(valid_neighbors)
            # for v_n in valid_neighbors:
            for v_n in tri['neighbors']:
                if v_n < 0:
                    continue
            
                if tris[v_n]['state'] > 0.0:
                    # x_ = 0.0 * tris[tri_i]['state'] + (0.9 * tris[v_n]['state'])
                    # x_ = 0.5 * tris[tri_i]['state'] + (0.5 * tris[v_n]['state'])
                    # coupling = 0.05
                    coupling = 0.2
                    transfer = coupling * tris[v_n]['state']
                    y_ += transfer
                    tris[v_n]['state'] -= transfer # coupling * tris[v_n]['state']

        
            # tris[tri_i]['state'] *= 0.5
            # x_ = np.tanh(x_)
            # x_ = np.sqrt(x_)
        
            # x_ = 0.92 * x_
        
            # decay activation
            # tris[tri_i]['state'] *= 0.98
            tris[tri_i]['state'] *= 0.8
            
            # add inputs
            tris[tri_i]['state'] += y_
            
            # output transfer function
            # tris[tri_i]['state_o'] = np.log(tris[tri_i]['state'] + 1) * 2
            tris[tri_i]['state_o'] = np.tanh(tris[tri_i]['state'] * 5)
            
class smnode(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(smnode, self).__init__()

        self.isrunning = True
        
        # assert mesh is not None, "Need to supply mesh argument" 
        # assert tris is not None, "Need to supply tris argument" 
        # self.mesh = mesh
        # self.tris = tris
        self.cnt = 0
        self.smid = 0
        self.density = np.random.uniform(0, 0.05)
        self.freq = 1/self.density
        self.neighbors = []
        self.inputs = {}
        self.state = np.zeros((1,1))
        self.outputs = {
            'state_o': np.zeros_like(self.state)
        }
        for k in ['smid', 'density', 'freq', 'neighbors']:
            if k in kwargs:
                setattr(self, k, kwargs[k])
        
    def run(self):
        while self.isrunning:
            # print('smnode {0}'.format(self.smid))
            # todo:
            # - read inputs
            # - compute output
            
            # self.mesh_update_state(self.cnt, self.mesh, self.tris, self.density)
            self.update()
            self.cnt += 1
            time.sleep(1/20.)

    def update(self):
        print('smnode-{0}.update {1}'.format(self.smid, self.inputs))

        x_ = self.state
        # print(tri_i, x_)
        y_ = np.zeros_like(x_)
        # periodic activation
        
        # y_ += 0.05 * np.sin((cnt/20.0) * tris[tri_i]['freq'] * 2 * np.pi)
        # y_ += 0.3 * np.sin(cnt/1000.0)**2
        
        # if tri_i == 0 and cnt % 100 == 0:
        # if tri_i == 0 and np.random.uniform() < event_density:
        if np.random.uniform() < self.density:
            # print('refreshing state')
            # tris[tri_i]['state'] = 1.0
            y_ += 2.0 + np.random.uniform(0, 2)
            # tris[tri_i]['state'] = x_

        
        # print(tri_i, neighbors)
        
        # print(valid_neighbors)
        for input_n in self.inputs:
            
            if self.inputs[input_n] > 0.0:
                # x_ = 0.0 * tris[tri_i]['state'] + (0.9 * tris[v_n]['state'])
                # x_ = 0.5 * tris[tri_i]['state'] + (0.5 * tris[v_n]['state'])
                # coupling = 0.05
                coupling = 0.2
                transfer = coupling * self.inputs[input_n]
                y_ += transfer
                # tris[v_n]['state'] -= transfer # coupling * tris[v_n]['state']
                
        
        # tris[tri_i]['state'] *= 0.5
        # x_ = np.tanh(x_)
        # x_ = np.sqrt(x_)
        
        # x_ = 0.92 * x_
        
        # decay activation
        # tris[tri_i]['state'] *= 0.98
        self.state *= 0.8
        
        # add inputs
        self.state += y_
            
        # output transfer function
        # tris[tri_i]['state_o'] = np.log(tris[tri_i]['state'] + 1) * 2
        self.outputs['state_o'] = np.tanh(self.state * 5)
            
class meshTrimesh(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(meshTrimesh, self).__init__()
        # create a mesh with mesh generation parameters
        self.mesh = self.make_mesh_triangle_trimesh(**kwargs)

        # compute the neighbors for each cell
        self.valid_neighbors_all = self.mesh_get_neighbors_trimesh(self.mesh)
        # print('valid_neighbors_all = {0}'.format(self.valid_neighbors_all))
        
        # extend the mesh with an attribute dictionary
        self.tris = self.mesh_extended_trimesh(self.mesh)

        self.coupling = 0.2
        self.isrunning = True
        self.cnt = 0
        
    def run(self):
        while self.isrunning:
            self.update()
            self.cnt += 1
            time.sleep(1/20.)

    def update(self):
        # todo
        # - loop over neighbors
        for nbrs in self.mesh.face_adjacency:
            # - populate node inputs with external values
            # self.mesh.face_attributes['smnode'][nbrs[0]].inputs['n{0}'.format(nbrs[1])] = self.mesh.face_attributes['state_o'][nbrs[1]]
            # self.mesh.face_attributes['smnode'][nbrs[1]].inputs['n{0}'.format(nbrs[0])] = self.mesh.face_attributes['state_o'][nbrs[0]]
            self.mesh.face_attributes['smnode'][nbrs[0]].inputs['n{0}'.format(nbrs[1])] = self.coupling * self.mesh.face_attributes['smnode'][nbrs[1]].state
            self.mesh.face_attributes['smnode'][nbrs[1]].state -= self.coupling * self.mesh.face_attributes['smnode'][nbrs[1]].state
            self.mesh.face_attributes['smnode'][nbrs[1]].inputs['n{0}'.format(nbrs[0])] = self.mesh.face_attributes['smnode'][nbrs[0]].state
            self.mesh.face_attributes['smnode'][nbrs[0]].state -= self.coupling * self.mesh.face_attributes['smnode'][nbrs[0]].state
        
    def make_mesh_triangle_trimesh(self, **params):
        """make_mesh_triangle_trimesh
        
        create mesh using trimesh.Trimesh
        """
        c = params['c']
        mesh_info = MeshInfo()

        # generate vertices and facets
        if params['obj'] == 'line':
            points, facets, faces = make_vertex_facets_line(params)
        elif params['obj'] == 'hexagon':
            points, facets, faces = make_vertex_facets_hexagon(params)
        elif params['obj'] == 'rect':
            points, facets, faces = make_vertex_facets_rect_trimesh(params)
        
        # print('points = {0}\nfacets = {1}'.format(pformat(points), pformat(facets)))

        # mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        #                        faces=[[0, 1, 2]])

        # face_attributes = {
        #     'color': len(faces) * [0],
        #     'state': [],
        #     'freq': [],
        # }
        # print('face_attributes = {0}'.format(face_attributes))
        
        mesh = trimesh.Trimesh(vertices=points, faces=faces)

        # print('mesh.edges = {0}'.format(mesh.edges))
        
        # writing objects
        # mesh.write_vtk("trigrid.vtk")
        # f = open('trigrid.pkl', 'wb')
        # pickle.dump(mesh, f)
        # f.close()
        # joblib.dump(mesh, 'trigrid.pkl')
        # sys.exit()
        return mesh
        
    def mesh_get_neighbors_trimesh(self, mesh):
        # nbrs = mesh.neighbors
        nbrs = mesh.face_adjacency
        valid_neighbors_all = []
        for nbr in nbrs:
            valid_neighbors_all.append([_ for _ in nbr if _ > -1])
        return(valid_neighbors_all)

    def mesh_extended_trimesh(self, mesh):
        """mesh_extended_trimesh

        create mesh extended with face attributes
        """
        # print('mesh.vertices = {0}'.format(pformat(mesh.vertices)))
        # print('mesh.faces = {0}'.format(pformat(mesh.faces)))
        # print('mesh.face_adjacency = {0}'.format(pformat(mesh.face_adjacency)))
    
        # plt.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh_tris)
        # plt.aspect(1)
        # plt.show()

        # create list of attribute dictionaries
        tris = []
        # loop over faces
        for i, tri_ in enumerate(mesh.faces):
            # print('tri_ = {0}'.format(tri_))
    
            tri_l = []
            for tri_vert in tri_:
                vert = mesh.vertices[tri_vert].tolist()
                # print('tri_vert = {0}'.format(vert))
                # print()
                tri_l.append(vert)

            tris.append({
                'vertices': tri_l,
                'neighbors': [], # list(mesh.face_adjacency[i]),
                'color': np.random.uniform(0, 1, (3,)),
                # 'color': np.array([0.7, 0.2, 0.1]),
                'freq': np.random.uniform(0.05, 0.2),
                'state': 0., # np.random.uniform(0, 1)
                'state_o': 0., # np.random.uniform(0, 1)
                # 'inputs': {}, # np.random.uniform(0, 1)
            })
            
        for nbr in mesh.face_adjacency:
            tris[nbr[0]]['neighbors'].append(nbr[1])
            # tris[nbr[0]]['inputs']['n{0}'.format(nbr[1])] = 0.
            tris[nbr[1]]['neighbors'].append(nbr[0])
            # tris[nbr[1]]['inputs']['n{0}'.format(nbr[0])] = 0.

        # update mesh face_attributes
        mesh.face_attributes.update(ld2dl(tris))
        
        # tris is list of attribute dictionaries
        # print('tris = {0}'.format(pformat(tris)))
        # want dictionary of attributes with list data
        # print('mesh.face_attributes = {0}'.format(pformat(mesh.face_attributes)))
        return tris

    def Cube(self, cnt, mesh, tris, valid_neighbors_all):
        mdir = 1.0
        mesh_points = np.array(mesh.vertices)
        mesh_tris = np.array(mesh.faces)
    
        glBegin(GL_TRIANGLES)
        for i, face in enumerate(mesh.faces):
            v_color = mesh.face_attributes['color'][i]
            # hack
            # mesh.face_attributes['state_o'][i] = tris[i]['state_o']
            # v_state_o = mesh.face_attributes['state_o'][i]
            v_state_o = mesh.face_attributes['smnode'][i].outputs['state_o']
            glColor3fv(v_color * v_state_o)
            # draw face vertices, taken directly from mesh.vertices
            for vert in mesh.vertices[face]:
                # print(vert)
                glVertex3fv(vert)
        glEnd()

        # draw lines
        glBegin(GL_LINES)
        for edge in mesh.edges:
            glColor3f(1, 1, 1)
            # print(edge)
            # glScalef(10.0, 10.0, 10.0)
            # glBegin(GL_TRIANGLES)
            # for edge in edges:
            for vertex in edge:
                # print(vertex, mesh_points[vertex])
                glVertex3fv(mesh_points[vertex].tolist())
        glEnd()
        
        # glBegin(GL_POINTS)
        # glColor3fv([1, 0, 0])
        # glVertex3fv([0, 0, 0])
        # glEnd()

class meshMeshpy(object):
    def __init__(self, *args, **kwargs):
        super(meshMeshpy, self).__init__()
        # create a mesh with mesh generation parameters
        self.mesh = self.make_mesh_triangle_meshpy(**kwargs)

        # compute the neighbors for each cell
        self.valid_neighbors_all = self.mesh_get_neighbors_meshpy(self.mesh)
        # print('valid_neighbors_all = {0}'.format(self.valid_neighbors_all))
        
        # extend the mesh with an attribute dictionary
        self.tris = self.mesh_extended_meshpy(self.mesh)
        
    def make_mesh_triangle_meshpy(self, **params):
        """make_mesh_meshpy_triangle
        
        create mesh using meshpy.triangle.MeshInfo
        """
        c = params['c']
        mesh_info = MeshInfo()
        
        # generate vertices and facets
        if params['obj'] == 'line':
            points, facets, faces = make_vertex_facets_line(params)
        elif params['obj'] == 'hexagon':
            points, facets, faces = make_vertex_facets_hexagon(params)
        elif params['obj'] == 'rect':
            points, facets = make_vertex_facets_rect(params)
    
        # print('points = {0}\nfacets = {1}'.format(pformat(points), pformat(facets)))
        # print('mesh_info.unit = {0}'.format(mesh_info.unit))
        
        # copy points data into mesh
        mesh_info.set_points(points)

        # copy facets data into mesh
        mesh_info.set_facets(facets)
    
        # build the mesh
        mesh = build(mesh_info)

        # writing objects
        # mesh.write_vtk("trigrid.vtk")
        # f = open('trigrid.pkl', 'wb')
        # pickle.dump(mesh, f)
        # f.close()
        # joblib.dump(mesh, 'trigrid.pkl')
        # sys.exit()
        return mesh

    def mesh_get_neighbors_meshpy(self, mesh):
        nbrs = mesh.neighbors
        valid_neighbors_all = []
        for nbr in nbrs:
            valid_neighbors_all.append([_ for _ in nbr if _ > -1])
        return(valid_neighbors_all)

    def mesh_extended_meshpy(self, mesh):
        mesh_points = np.array(mesh.points)
        mesh_tris = np.array(mesh.elements)
        mesh_neighbors = np.array(mesh.neighbors)

        # print('mesh_tris = {0}'.format(list(mesh_tris)))
        # print('neighbors = {0}'.format(list(mesh.neighbors)))
        
        # plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
        # plt.aspect(1)
        # plt.show()

        # for
        tris = []
        # tris_d = []
        for i, tri_ in enumerate(mesh_tris):
            # print('tri_ = {0}'.format(tri_))
    
            tri_l = []
            for tri_vert in tri_:
                vert = mesh_points[tri_vert].tolist()
                vert += [0]
                # print('tri_vert = {0}'.format(vert))
                # print()
                tri_l.append(vert)
            # tris.append(tri_l)
            tris.append({
                'vertices': tri_l,
                'neighbors': list(mesh_neighbors[i]),
                'color': np.random.uniform(0, 1, (3,)),
                # 'color': [0.5, 0.2, 0.3],
                'state': 0., # np.random.uniform(0, 1)
                'state_o': 0., # np.random.uniform(0, 1)
            })

        # print('tris = {0}'.format(tris))
        return tris

    def Cube(self, cnt, mesh, tris, valid_neighbors_all):
        mdir = 1.0
        mesh_points = np.array(mesh.points)
        mesh_tris = np.array(mesh.elements)
    
        glBegin(GL_TRIANGLES)
        # glColor3fv(vertcolors[j] * vertstate[j])
        # glVertex3fv(vertices[vertex_i])
        # glColor3fv([1, 1, 1])
        for i, tri in enumerate(tris):

            # if i > 0:
            #     tris[i]['state'] += 0.05 * tris[i-1]['state']
            # if i < (len(tris) - 1):
            #     tris[i]['state'] -= 0.13 * tris[i+1]['state'] + np.random.uniform(-1e-2, 1e-2)

            # glColor3fv([np.random.uniform(), 1, 1])
            glColor3fv(tris[i]['color'] * tris[i]['state'])
        
            for vert in tri['vertices']:
                # print(vert)
                glVertex3fv(vert)
        glEnd()

        # draw lines
        glBegin(GL_LINES)
        for edge in mesh.facets:
            glColor3f(1, 1, 1)
            # print(edge)
            # glScalef(10.0, 10.0, 10.0)
            # glBegin(GL_TRIANGLES)
            # for edge in edges:
            for vertex in edge:
                # print(vertex, mesh_points[vertex])
                glVertex3fv(mesh_points[vertex].tolist() + [0])
        glEnd()
        
        # glBegin(GL_POINTS)
        # glColor3fv([1, 0, 0])
        # glVertex3fv([0, 0, 0])
        # glEnd()

def get_params(obj='line', c=1, dim=3):
    hc1 = np.sin(np.deg2rad(60)) * c
    r_i = hc1/3
    r_o = 2 * r_i
    params = {
        'obj': obj,
        'c': c,
        'dim': dim,
        'hc1': hc1,
        'hc2': np.sqrt(3)/2 * c,
        'r_i': r_i,
        'r_o': r_o,
        'c_2': ((c/2) * r_o) / hc1,
    }
    return params

def main(args):
    """meshgrid.main

    create a mesh of computation nodes

    - create mesh (generate, load)
    - populate with attributes
    - populate with threaded nodes
    - render mesh based on fixed set of attributes
    """
    
    # get mesh generation parameters
    if args.meshlib == 'trimesh':
        dim = 3
        meshClass = meshTrimesh
    elif args.meshlib == 'meshpy':
        dim = 2
        meshClass = meshMeshpy
        
    params = get_params(obj=args.mode, c=1, dim=dim)
    # print('params = {0}'.format(pformat(params)))
    mesh = meshClass(**params)

    # populate nodes
    mesh.mesh.face_attributes['smnode'] = []
    for i, face in enumerate(mesh.mesh.faces):
        mesh.mesh.face_attributes['smnode'].append(smnode(smid=i))
        mesh.mesh.face_attributes['smnode'][-1].start()

    # # create state update thread
    # ru = runUpdate(mesh.mesh, mesh.tris, args.density)
    # # start state update thread
    # ru.start()

    # start mesh update thread
    mesh.start()
    
    # initialize pygame and OpenGL
    pygame.init()
    # display = (800,600)
    display = (1200, 900)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    gluPerspective(60, (display[0]/display[1]), 0.1, 50.0)
    
    # hexagon
    if args.mode == 'hexagon':
        glTranslatef(-0.0, 0.0, -5)
    # line
    elif args.mode == 'line':
        glTranslatef(-6.0, 0.0, -10)
    # rect
    elif args.mode == 'rect':
        glTranslatef(0, 0, -10)

    # glScalef(2.0, 2.0, 2.0)
    glScalef(3.0, 3.0, 3.0)

    # start main loop
    running = True
    cnt = 0
    while running:
        # event handling
        for event in pygame.event.get():
            # quit event
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            # if event.type == pygame.QUIT:
                print('Got QUIT event')
                running = False
                ru.isrunning = False
                ru.join()
                # terminate mesh update thread
                mesh.isrunning = False
                mesh.join()
                # join smnode threads from mesh
                for i, face in enumerate(mesh.mesh.faces):
                    mesh.mesh.face_attributes['smnode'][i].isrunning = False
                    mesh.mesh.face_attributes['smnode'][i].join()
                pygame.quit()
                quit()

        # glRotatef(1, 3, 3, 3)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # render function on mesh
        mesh.Cube(cnt, mesh.mesh, mesh.tris, mesh.valid_neighbors_all)
        
        # bookkeeping
        cnt += 1
        pygame.display.flip()
        pygame.time.wait(20)
        # pygame.time.wait(50)

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--density', type=float, default=0.1, help='Density for random node activity [0.1]')
    parser.add_argument('-m', '--mode', type=str, default='hexagon', help='Mesh mode [hexagon] (hexagon, line, rect)')
    parser.add_argument('-l', '--meshlib', type=str, default='trimesh', help='Which meshlib to use [trimesh] (trimesh, meshpy)')

    args = parser.parse_args()
    
    main(args)
