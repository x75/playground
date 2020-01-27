"""triangular grid

see trimesh, networkx
 - load mesh
 - get triangle graph with neighbors
 - create the mesh as a graph
 - compute cells, render cell state onto mesh

"""

from __future__ import division
from __future__ import absolute_import

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
from six.moves import range


c = 1
hc1 = np.sin(np.deg2rad(60)) * c
hc2 = np.sqrt(3)/2 * c
r_i = hc1/3
r_o = 2 * r_i
c_2 = ((c/2) * r_o) / hc1
# r_o = np.sin(np.deg2rad(30)) * 1

print(hc1, hc2, r_i, r_o)

# def get_trigrid(num_x, num_y, c):
#     hc = np.sqrt(3)/2 * c
#     grid = []
#     # np.cos(np.deg2rad(30)) * r_o, np.sin(np.deg2rad(30)) * r_o
#     c_ = np.cos(np.deg2rad(30)) * r_o
#     h_ = np.sin(np.deg2rad(30)) * r_o
#     for i_y in range(num_y):
#         col = []
#         if i_y % 2 == 0:
#             off_x = 0
#         else:
#             off_x = c/2
            
#         for i_x in range(num_x):
#             if i_x % 2 == 0:
#                 off_y = h_
#             else:
#                 off_y = 0

#             col.append([i_x * c_ - off_x, i_y * hc + off_y])
#         grid.append(col)
#     return grid

# trigrid = get_trigrid(4, 2, 1)
# print('trigrid = {0}'.format(trigrid))

# # define triangle / base shape
# tri2d = np.array([
#     [-0.5, 0],
#     [0.5, 0],
#     [0, np.sin(np.deg2rad(60)) * 1]
# ])


# tri3d = [
#     [-0.5, -r_i, 0],
#     [0.5, -r_i, 0],
#     [0, hc1-r_i, 0]
# ]

# rot = R.from_euler('z', -60, degrees=True).as_matrix()
# print('rot = {0}'.format(rot))
# # Mrot = r.as_matrix()

# # iterate base shape to create grid
# x = np.array(tri3d)
# # x_tr = np.array([np.cos(np.deg2rad(30)) * r_o, np.sin(np.deg2rad(30)) * r_o, 0])
# x_tr = np.array([np.cos(np.deg2rad(30)) * r_o, np.sin(np.deg2rad(30)) * r_o, 0])

# print('x = {0}'.format(x))
# tris = []
# # for i in range(1):
# #     x = np.dot(rot, x.T).T
# #     print('x = {0}'.format(x))
# #     # tris.append(x + (i * hc1/2))
# #     tris.append(x + x_tr)

# for i, trigrid_col in enumerate(trigrid):
#     for j, trigrid_row in enumerate(trigrid_col):
#         x_tr = np.array(trigrid_row + [0])
#         if j % 2 == 0:
#             x_ = np.dot(rot, x.T).T
#         else:
#             x_ = x
#         tris.append(x_ + x_tr)

# # rotate / translate


# use meshpy
from meshpy.triangle import MeshInfo, build

mesh_info = MeshInfo()

points = [(0, 0)]
for ang in [0, 60, 120, 180, 240, 300]:
    points.append((np.cos(np.deg2rad(ang)) * 1, np.sin(np.deg2rad(ang)) * 1))
facets = [
    [0,1],
    [0,2],
    [0,3],
    [0,4],
    [0,5],
    [0,6],
    [1,2],
    [2,3],
    [3,4],
    [4,5],
    [5,6],
    [6,1],
    # [2,3],
    # [3,4],
    # [4,5],
    # [5,6],
    # [6,4],
]


# points = [(0,0)]
# for trans in [0, 1, 2, 3, 4]:
#     angs = [0, 60]
#     # if trans > 0:
#     #     angs += [120]
        
#     for ang in angs:
#         points.append((np.cos(np.deg2rad(ang)) * 1 + trans, np.sin(np.deg2rad(ang)) * 1))

# print('points = {0}'.format(points))

# facets = [
#     [0, 1], [0, 2], [1,2],
#     [1, 2], [1, 4], [2,4],
#     [1, 3], [1, 4], [3,4],
#     [3, 4], [3, 6], [4,6],
#     [3, 5], [3, 6], [5,6],
#     [5, 6], [5, 8], [6,8],
#     [5, 7], [5, 8], [7,8],
# ]

# mesh_info.set_points([
#     # (0, 0), (1, 0), (0.5, hc1),
#     (0, 0),
#     (1, 0), (1, 1), (0, 1),
#     # (2, 0), (1, 2), (0, 2),
#     ])

print(points)
mesh_info.set_points(points)

mesh_info.set_facets(facets)

mesh = build(mesh_info)
# mesh.write_vtk("test.vtk")

mesh_points = np.array(mesh.points)
mesh_tris = np.array(mesh.elements)
print('mesh_tris = {0}'.format(list(mesh_tris)))

print('neighbors = {0}'.format(list(mesh.neighbors)))

plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
# plt.aspect(1)
# plt.show()

# for
tris = []
tris_d = []
for tri_ in mesh_tris:
    print('tri_ = {0}'.format(tri_))
    
    tri_l = []
    for tri_vert in tri_:
        vert = mesh_points[tri_vert].tolist()
        vert += [0]
        print('tri_vert = {0}'.format(vert))
        # print()
        tri_l.append(vert)
    tris.append(tri_l)
    tris_d.append({
        'color': np.random.uniform(0, 1, (3,)),
        'state': np.random.uniform(0, 1)
    })

print(tris)
        
def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]

def main_triangle():
    points = [(1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1), (1, 0)]
    facets = round_trip_connect(0, len(points)-1)

    circ_start = len(points)
    points.extend(
            (3 * np.cos(angle), 3 * np.sin(angle))
            for angle in np.linspace(0, 2*np.pi, 30, endpoint=False))

    facets.extend(round_trip_connect(circ_start, len(points)-1))

    def needs_refinement(vertices, area):
        bary = np.sum(np.array(vertices), axis=0)/3
        max_area = 0.001 + (la.norm(bary, np.inf)-1)*0.01
        return bool(area > max_area)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_holes([(0, 0)])
    info.set_facets(facets)

    mesh = triangle.build(info, refinement_func=needs_refinement)

    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)

    # import matplotlib.pyplot as pt
    pt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris)
    pt.show()

# if __name__ == "__main__":
#     main()



def Cube(cnt):
    mdir = 1.0

    for tri_i, tri in enumerate(tris):
        # x_ = 0.9 * tris_d[tri_i]['state']
        # tris_d[tri_i]['state'] = x_
        pass
        
    for tri_i, neighbors in enumerate(mesh.neighbors):
        x_ = 0.99 * tris_d[tri_i]['state']
        # tris_d[tri_i]['state'] = x_
        
        # periodic activation
        if tri_i == 0 and cnt % 100 == 0:
            print('refreshing state')
            # tris_d[tri_i]['state'] = 1.0
            x_ = 1.0
            tris_d[tri_i]['state'] = x_
        
        # print(tri_i, neighbors)

        valid_neighbors = [_ for _ in neighbors if _ > -1]
        
        # # print(valid_neighbors)
        # for v_n in valid_neighbors:
            
        #     if tris_d[v_n]['state'] > 0.1:
        #         x_ = 0.0 * tris_d[tri_i]['state'] + (0.9 * tris_d[v_n]['state'])
                # x_ = 0.5 * tris_d[tri_i]['state'] + (0.5 * tris_d[valid_neighbors[0]]['state'])
        
        # hexagon circular rule
        if len(valid_neighbors) < 2:
            continue
        
        if tris_d[valid_neighbors[0]]['state'] > tris_d[tri_i]['state']:
            x_ = 0.5 * tris_d[tri_i]['state'] + (0.5 * tris_d[valid_neighbors[0]]['state'])
        elif tris_d[valid_neighbors[1]]['state'] > tris_d[tri_i]['state']:
            x_ = 0.5 * tris_d[tri_i]['state'] + (0.5 * tris_d[valid_neighbors[1]]['state'])
        else:
            x_ = 0.9 * tris_d[tri_i]['state']
        
        # tris_d[tri_i]['state'] = 1.01 * tris_d[tri_i]['state'] * (1 - tris_d[valid_neighbors[1]]['state']) * (1 - tris_d[valid_neighbors[0]]['state'])
        # tris_d[tri_i]['state'] = 1.01 * tris_d[tri_i]['state'] * (1 - tris_d[valid_neighbors[1]]['state']) * (1 - tris_d[valid_neighbors[0]]['state'])
        
        # if tris_d[valid_neighbors[0]]['state'] > 0.9:
        #     x_ = 0.1 * tris_d[tri_i]['state'] + (0.9 * tris_d[valid_neighbors[0]]['state'])
        # else:
        #     x_ = 0.96 * tris_d[tri_i]['state']

        # if tris_d[valid_neighbors[1]]['state'] > 0.8:
        #     x_ = 0.1 * tris_d[tri_i]['state'] + (0.9 * tris_d[valid_neighbors[1]]['state'])
        # else:
        #     x_ = 0.96 * tris_d[tri_i]['state']
            
        # x_ = (0.99 * tris_d[valid_neighbors[0]]['state'])
        # np.tanh(x_ * 2 - 1) / 2 + 0.5
        
        tris_d[tri_i]['state'] = x_

        # for n_i, neighbor in enumerate(neighbors):
        #     # print(n_i, neighbors)
        #     if n_i % 2 == 0:
        #         ndir = -1
        #     else:
        #         ndir = 1

        #     if neighbor > 0:
        #         # if tris_d[tri_i]['state'] > 0.9:
        #         #     mdir = -1.0
        #         # elif tris_d[tri_i]['state'] < 0.1:
        #         #     mdir = 1.0
        #         #     # tris_d[tri_i]['state'] += mdir * 0.01 * tris_d[n_i]['state']
                
        #         # tris_d[tri_i]['state'] += ndir * mdir * 0.05 * tris_d[n_i]['state']
        #         tris_d[tri_i]['state'] = 3.2 * tris_d[n_i]['state'] * (1 - tris_d[tri_i]['state'])
        # # tris_d[tri_i]['state'] = 3.2 * tris_d[tri_i]['state'] * (1 - tris_d[tri_i]['state'])
    
    glBegin(GL_TRIANGLES)
    # glColor3fv(vertcolors[j] * vertstate[j])
    # glVertex3fv(vertices[vertex_i])
    # glColor3fv([1, 1, 1])
    for i, tri in enumerate(tris):

        # if i > 0:
        #     tris_d[i]['state'] += 0.05 * tris_d[i-1]['state']
        # if i < (len(tris) - 1):
        #     tris_d[i]['state'] -= 0.13 * tris_d[i+1]['state'] + np.random.uniform(-1e-2, 1e-2)

        # glColor3fv([np.random.uniform(), 1, 1])
        glColor3fv(tris_d[i]['color'] * tris_d[i]['state'])
        
        for vert in tri:
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
    
def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    # gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    gluPerspective(60, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(-0.0, 0.0, -5)
    # glTranslatef(-6.0, 0.0, -5)
    # glTranslatef(-6.0, 0.0, -20)

    glScalef(3.0, 3.0, 3.0)
    
    running = True
    cnt = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            # if event.type == pygame.QUIT:
                print('Got QUIT event')
                running = False
                pygame.quit()
                quit()

        # glRotatef(1, 3, 3, 3)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube(cnt)
        cnt += 1
        pygame.display.flip()
        pygame.time.wait(20)
        # pygame.time.wait(50)


if __name__ == '__main__':
    main()
