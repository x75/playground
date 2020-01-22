import pygame
from pygame.locals import *
import trimesh

from OpenGL.GL import *
from OpenGL.GLU import *

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

from meshpy.tet import MeshInfo, build

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



# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/featuretype.STL')
# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/unit_sphere.STL')
mesh = trimesh.load_mesh('/home/src/python/trimesh/models/unit_cube.STL')
# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/origin_inside.STL')
# mesh = trimesh.load_mesh('/home/src/python/trimesh/models/soup.stl')
vertices = mesh.faces
edges = mesh.edges

import numpy as np
import scipy.spatial
import pylab
from itertools import combinations

data = np.random.uniform([-5, -5, -1], [5, 5, 1], (20,3))            # arbitrary 3D data set
tri = scipy.spatial.Delaunay( data[:,:2] ) # take the first two dimensions

print('tri.simplices = {0}'.format(tri.simplices.copy()))

pylab.triplot( data[:,0], data[:,1], tri.simplices.copy() )
pylab.plot( data[:,0], data[:,1], 'ro' ) ;
pylab.show()

vertices = data
edges = tri.simplices

vertcolors = []
vertstate = []

for edge in edges:
    vertcolors.append(np.random.uniform(0, 1, (3,)))
    vertstate.append(np.random.uniform(0, 1))

print('vertices = {0}\nedges = {1}'.format(vertices, edges))

def Cube():

    glBegin(GL_TRIANGLES)
    
    for i,edge in enumerate(edges):
        glColor3fv(vertcolors[i] * vertstate[i])
        for vertex in edge:
            glVertex3fv(vertices[vertex])
        if i > 0:
            vertstate[i] += 0.05 * vertstate[i-1]
        if i < (len(edges) - 1):
            vertstate[i] -= 0.13 * vertstate[i+1] + np.random.uniform(-1e-2, 1e-2)
        vertstate[i] = np.tanh(vertstate[i])
            
    glEnd()

    glBegin(GL_LINES)
    glColor3f(1, 1, 1)
    # glBegin(GL_TRIANGLES)
    for edge in edges:
        # for vertex in edge:
        #     glVertex3fv(vertices[vertex])
        for vertex in combinations(edge, 2):
            for line in vertex:
                glVertex3fv(vertices[line])
    glEnd()

    # for i, pt in enumerate(data):
    #     if np.random.uniform(0, 1) < 0.01:
    #         # data[i][0] += np.random.uniform(0, 1e-1)
    #         data[i][0] += np.sin(pt[1])
    #     if np.random.uniform(0, 1) < 0.01:
    #         # data[i][1] += np.random.uniform(0, 1e-1)
    #         data[i][0] += np.cos(pt[0])
        
    
def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    gluPerspective(60, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0,0.0, -5)
    # glTranslatef(-6.0, 0.0, -20)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # glRotatef(0.1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(20)


main()
