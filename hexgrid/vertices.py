import pygame
from pygame.locals import *
import trimesh

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import scipy.spatial
import pylab
from itertools import combinations

from scipy.spatial.transform import Rotation as R

from meshpy.tet import MeshInfo, build

# vertices = (
#     (1, -1, -1),
#     (1, 1, -1),
#     (-1, 1.3, -1),
#     (-1, -1, -1.7),
#     (1, -1, 1),
#     (1, 1, 1),
#     (-1, -1, 1),
#     (-1, 1, 1)
#     )

# edges = (
#     (0,1),
#     (0,3),
#     (0,4),
#     (2,1),
#     (2,3),
#     (2,7),
#     (6,3),
#     (6,4),
#     (6,7),
#     (5,1),
#     (5,4),
#     (5,7)
#     )

vertices = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    )

edges = (
    (1,2),
    (0,1),
    (0,2),
    )

mesh_info = MeshInfo()
mesh_info.set_points([
    (0,0,0), (2,0,0), (2,2,0), (0,2,0),
    (0,0,12), (2,0,12), (2,2,12), (0,2,12),
    ])
mesh_info.set_facets([
    [0,1,2,3],
    [4,5,6,7],
    [0,4,5,1],
    [1,5,6,2],
    [2,6,7,3],
    [3,7,4,0],
    ])
mesh = build(mesh_info)
# mesh.write_vtk("test.vtk")

# create large hexagon
# define equilateral triangle
triangle_2d = np.array([
    [-0.5, 0],
    [0.5, 0],
    [0, np.sin(np.deg2rad(60)) * 1]
])

triangle = [
    [-0.5, 0, 0],
    [0.5, 0, 0],
    [0, np.sin(np.deg2rad(60)) * 1, 0]
]
print('triangle = {0}'.format(triangle))

# mesh_info.set_points(triangle)
# mesh_info.set_facets([
#     [0,1,2]
# ])
# mesh = build(mesh_info)

trimesh.util.attach_to_log()

# Mrot = trimesh.transformations.rotation_matrix(np.deg2rad(60), [0, 1, 0])
# Mtrans = trimesh.transformations.translation_matrix([0.5, 0, 0])

# print('Mrot = {0}'.format(Mrot))

triverts = []
for vert in triangle:
    triverts.append(vert)

# r = R.from_quat([0, 0, np.sin(np.pi/4), np.cos(np.pi/4)])
r = R.from_euler('z', -60, degrees=True)

Mrot = r.as_matrix()
print('r = {0}\nMrot = {1}'.format(r, Mrot))

# triangle_ = np.array(triangle).T
# # triangle_ = triangle_ + np.array([[0.5, 0.0, 0.0]])
# triangle_ = np.dot(Mrot, triangle_).T
# print('triangle_ = {0}'.format(triangle_)) # , triangle_rot, triangle_trans)) # \ntriangle_rot = {1}\ntriangle_trans = {2}
# for vert in triangle_:
#     triverts.append(vert.tolist())

# mesh objects can be created from existing faces and vertex data
mesh = trimesh.Trimesh(
    # vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    # vertices=triangle,
    vertices=triverts,
    faces=[
        [0, 1, 2],
#        [3, 4, 5],
    ]
)


print("mesh = {0}".format(mesh))

vertices = mesh.vertices
faces = mesh.faces
edges = mesh.edges

# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/featuretype.STL')
# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/unit_sphere.STL')
# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/unit_cube.STL')
# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/origin_inside.STL')
# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/soup.stl')
# vertices = mesh.faces
# edges = mesh.edges

# # data = np.random.uniform([-5, -5, -1], [5, 5, 1], (30,3))            # arbitrary 3D data set
# data = np.array(mesh.faces).astype(float)
# # print(np.max(data))
# print(data.dtype)
# data /= np.max(data) # - np.array((150, 150, 0))
# data -= np.mean(data)
# data *= 10

# tri = scipy.spatial.Delaunay( data[:,:2] ) # take the first two dimensions

# print('tri.simplices = {0}'.format(tri.simplices.copy()))

# # pylab.triplot( data[:,0], data[:,1], tri.simplices.copy() )
# # pylab.plot( data[:,0], data[:,1], 'ro' ) ;
# # pylab.show()

# vertices = data
# edges = tri.simplices

vertcolors = []
vertstate = []

# for edge in edges:
for edge in range(10):
    vertcolors.append(np.random.uniform(0, 1, (3,)))
    vertstate.append(np.random.uniform(0, 1))

print('vertices = {0}\nedges = {1}'.format(vertices, edges))

def Cube():

    # glBegin(GL_TRIANGLES)
    
    # for i,edge in enumerate(edges):
    #     glColor3fv(vertcolors[i] * vertstate[i])
    #     for vertex in edge:
    #         glVertex3fv(vertices[vertex])
    #     if i > 0:
    #         vertstate[i] += 0.05 * vertstate[i-1]
    #     if i < (len(edges) - 1):
    #         vertstate[i] -= 0.13 * vertstate[i+1] + np.random.uniform(-1e-2, 1e-2)
    #     vertstate[i] = np.tanh(vertstate[i])
            
    # glEnd()

    # glBegin(GL_LINES)
    # glColor3f(1, 1, 1)
    # # glScalef(10.0, 10.0, 10.0)
    # # glBegin(GL_TRIANGLES)
    # for edge in edges:
    #     # print(edge)
    #     # for vertex in edge:
    #     #     glVertex3fv(vertices[vertex])
    #     for vertex in combinations(edge, 2):
    #         for line in vertex:
    #             glVertex3fv(vertices[line])
    # glEnd()

    glPushMatrix()
    glMatrixMode(GL_MODELVIEW)
    glScalef(2.0, 2.0, 2.0)
    
    hc = np.sin(np.deg2rad(60))
    for j, segment in enumerate(range(6)):
        # glTranslatef(1.0, np.sin(np.deg2rad(-60)) * 0.5, 0.0)
        glRotatef(-60, 0, 0, 1)
        # glTranslatef(0.5, 0.5 * hc, 0.0)
        glTranslatef(0.25, -0.5 * hc, 0.0)
        # glRotatef(-60, 0, 0, 1)
        glBegin(GL_TRIANGLES)
        glColor3fv(vertcolors[j] * vertstate[j])
        
        if j > 0:
            vertstate[j] += 0.05 * vertstate[j-1]
        if j < (len(edges) - 1):
            vertstate[j] -= 0.13 * vertstate[j+1] + np.random.uniform(-1e-2, 1e-2)

        for i, face in enumerate(faces):
            # print('face = {0}'.format(face))
            for vertex_i in face:
                glVertex3fv(vertices[vertex_i])
        glEnd()
    
        glBegin(GL_LINES)
        glColor3f(1, 1, 1)
        # glScalef(10.0, 10.0, 10.0)
        # glBegin(GL_TRIANGLES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()
    
    glPopMatrix()
    
    # for i, pt in enumerate(data):
    #     if np.random.uniform(0, 1) < 0.1:
    #         data[i][0] += np.random.uniform(0, 1e-1)
    #         # data[i][0] = np.tanh(data[i][0])
    #         # data[i] = np.tanh(data[i] + 0.1 * np.sin(pt)) * 1
    #     if np.random.uniform(0, 1) < 0.1:
    #         data[i][1] += np.random.uniform(0, 1e-1)
    #         # data[i][1] = np.tanh(data[i][1])
    #         # data[i] = np.tanh(data[i] + 0.1 * np.cos(pt)) * 1
    
def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    gluPerspective(60, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0,0.0, -5)
    # glTranslatef(-6.0, 0.0, -20)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('Got QUIT event')
                running = False
                pygame.quit()
                quit()

        # glRotatef(1, 3, 3, 3)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(20)


main()
