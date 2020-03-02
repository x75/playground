import argparse, time, threading, queue
import pygame
from pygame.locals import *

import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *

import trimesh

from oscpy.server import OSCThreadServer

# message queue
# 1. setPerspective
# 2. glTranslate
# 3. glScalef
import zmq

from oscsrv import OSCsrv

class ctrlZmq(threading.Thread):
    def __init__(self):
        super(ctrlZmq, self).__init__()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:5555")
        self.isrunning = True

    def run(self):
        while self.isrunning:
            #  Wait for next request from client
            try:
                message = self.socket.recv(flags=zmq.NOBLOCK)
                print("Received request: %s" % message)
            except Exception as e:
                print("Receive timeout: %s" % e)

            #  Do some 'work'
            time.sleep(0.1)

            #  Send reply back to client
            try:
                self.socket.send(b"World")
            except Exception as e:
                pass
        print('leave ctrlZmq.run')
        self.socket.close()

class ctrlOSC(object):
    def __init__(self):
        super(ctrlOSC, self).__init__()
        # self.isrunning = True
        self.osc = OSCThreadServer()
        self.sock = self.osc.listen(address='0.0.0.0', port=8000, default=True)
        self.osc.bind(b'/address', self.callback)
        
    # def run(self):
    #     while self.isrunning:

    # @osc.address(b'/address')
    def callback(self, values, *args):
        print("got values: {}, {}, {}".format(type(values), values, args))

# threading.Thread
class trigridDisplay(object):
    def __init__(self):
        super(trigridDisplay, self).__init__()
        self.isinit = False
        # initialize pygame and OpenGL
        pygame.init()
        # display = (800,600)
        self.display = (1200, 900)
        pygame.display.set_mode(self.display, DOUBLEBUF|OPENGL)

        # width = 1200
        # height = 900
        # glViewport(0, 0, width, height)
        # glMatrixMode(GL_PROJECTION)
        # glLoadIdentity()
        # # perspective: camera opening angle y (height), ratio x : y, clip near, clip far
        # gluPerspective(45, 1.0*width/height, 0.1, 100.0)
        # glMatrixMode(GL_MODELVIEW)
        # glLoadIdentity()
        
        # self.resize((640, 480))

        self.setPerspective()
        glTranslatef(-0.0, 0.0, -5)
        
        self.isrunning = False
        
        # self.ctrl = ctrlZmq()
        # self.ctrl.start()
        
        # self.ctrl = ctrlOSC()        
        # self.ctrl.start()

        self.qu = queue.Queue(maxsize=10)
        self.ctrl = OSCsrv(queue=self.qu)
        # self.ctrl.start()
        self.mesh = None
        self.isinit = True
        
    def setPerspective(self):
        gluPerspective(60, (self.display[0]/self.display[1]), 0.1, 50.0)

    def setTranslate(self, trans):
        glTranslatef(trans[0], trans[1], trans[2])
        # pass

    def setScale(self, scale):
        glScalef(scale[0], scale[1], scale[2])
        # pass

    def renderVert(self, vert):
        print('renderVert {0}'.format(vert))
        # glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glBegin(GL_TRIANGLES)
        
        color = np.array(vert[:3])
        state = vert[3]
        colorstate = color * state
        glColor3fv(colorstate)
        # for i, v in enumerate(vert):
        for i in range(4, len(vert), 3):
            if i < 4:
                continue
            vert_i = vert[i:i+3]
            # print('renderVert {0}'.format(vert_i))
            glVertex3fv(vert_i)

        glEnd()
        # pass
        
    def loadmesh(self, meshfile):
        try:
            meshfile = 'trigrid-mesh.json'
            self.mesh = trimesh.load_mesh(meshfile)
            self.mesh.face_attributes['color'] = [np.random.uniform(0, 1, (3, )) for _ in range(len(self.mesh.faces))]
            print('mesh loaded', meshfile)
        except Exception as e:
            print('loadmesh failed', e)
        
    def facecolor(self, facecolor):
        if self.mesh is None:
            return
        self.mesh.face_attributes['color'][int(facecolor[0])] = facecolor[1:]
        
    def run(self):
        # print('enter trigridDisplay.run')
        self.isrunning = True
        cnt = 0
        # while self.isrunning:
        if True:
            # print('enter trigridDisplay.run loop {0}'.format(cnt))

            while self.qu.qsize() > 0:
                qud = self.qu.get()
                if qud is not None:
                    # print('qud {0}'.format(qud))
                    if qud[0] == '/translate':
                        self.setTranslate(qud[1])
                    elif qud[0] == '/scale':
                        self.setScale(qud[1])
                    elif qud[0] == '/vert':
                        self.renderVert(qud[1])
                    elif qud[0] == '/load':
                        self.loadmesh(qud[1])
                    elif qud[0] == '/facecolor':
                        self.facecolor(qud[1])
            
            # event handling
            for event in pygame.event.get():
                # quit event
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                # if event.type == pygame.QUIT:
                    print('Got QUIT event')
                    self.isrunning = False
                    self.ctrl.isrunning = False
                    print('waiting ctrl join')
                    # self.ctrl.join()
                    print('ctrl joined')
                    # ru.isrunning = False
                    # ru.join()
                    # terminate mesh update thread
                    # mesh.isrunning = False
                    # mesh.join()
                    # join smnode threads from mesh
                    # for i, face in enumerate(mesh.mesh.faces):
                    #     mesh.mesh.face_attributes['smnode'][i].isrunning = False
                    #     mesh.mesh.face_attributes['smnode'][i].join()
                    pygame.quit()
                    quit()


            self.rendermesh()
            # # glRotatef(1, 3, 3, 3)
            
            # glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

            # glBegin(GL_TRIANGLES)
            # glColor3fv([0.8, 0.8, 0.8])
            # glVertex3fv([0.0, 0, 0])
            # glVertex3fv([0, 0.1, 0])
            # glVertex3fv([0.1, 0, 0])
            # glEnd()

            # render function on mesh
            # mesh.Cube(cnt, mesh.mesh, mesh.tris, mesh.valid_neighbors_all)
        
            # bookkeeping
            cnt += 1
            pygame.display.flip()
            # pygame.time.wait(20)
            pygame.time.wait(40)

    def rendermesh(self):
        if self.mesh is None:
            return
        
        glBegin(GL_TRIANGLES)
        
        for i, face in enumerate(self.mesh.faces):


            try:
                facecol = self.mesh.face_attributes['color'][i]
            except:
                facecol = [.8, .8, .8]
            glColor3fv(facecol)

            # print('face colors', i, facecol)
        
            for vert in self.mesh.vertices[face]:
                # print(vert)
                glVertex3fv(vert)
        glEnd()

        # draw lines
        glBegin(GL_LINES)
        for edge in self.mesh.edges:
            glColor3f(1, 1, 1)
            # print(edge)
            # glScalef(10.0, 10.0, 10.0)
            # glBegin(GL_TRIANGLES)
            # for edge in edges:
            for vertex in edge:
                # print(vertex, mesh_points[vertex])
                glVertex3fv(self.mesh.vertices[vertex].tolist())
        glEnd()
        
            
def main(args):

    d = trigridDisplay()
    while not d.isinit:
        time.sleep(0.1)
        
    # d.start()
    isrunning = True
    while isrunning:
        d.run()
        # time.sleep(1.0)
    
if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    
    main(args)
