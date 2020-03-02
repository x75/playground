import liblo
import threading

class OSCsrv(object):
    def __init__(self, port=1234, queue=None):
        self.server = None
        self.port = port
        self.queue = queue
        try:
            self.server = liblo.Server(self.port)
            self.isrunning = True
        except liblo.ServerError as err:
            print(err)
            self.isrunning = False
            # sys.exit()
            
        # register method taking an int and a float
        # self.server.add_method("/mfcc", 'f' * 38, self.cb_mfcc)
        # self.server.add_method("/beat", 'f' * 3, self.cb_beat)
        self.server.add_method("/address", 'if', self.cb_address)
        self.server.add_method("/translate", 'fff', self.cb_translate)
        self.server.add_method("/scale", 'fff', self.cb_scale)
        self.server.add_method("/vert", 'fffffffffffff', self.cb_vert)
        self.server.add_method("/load", 'i', self.cb_load)
        self.server.add_method("/facecolor", 'ifff', self.cb_facecolor)
        self.callbacks = []

        self.st = threading.Thread( target = self.run )
        self.st.start()

    def cb_address(self, path, args):
        # print('received args {0}'.format(args))
        self.address = args
        self.queue.put((path, args))
        
    def cb_translate(self, path, args):
        print('cb_translate received args {0}'.format(args))
        self.translate = args
        self.queue.put((path, args))
        
    def cb_scale(self, path, args):
        print('cb_scale received args {0}'.format(args))
        self.scale = args
        self.queue.put((path, args))
        
    def cb_vert(self, path, args):
        # print('received args {0}'.format(args))
        self.vert = args
        self.queue.put((path, args))
        
    def cb_facecolor(self, path, args):
        # print('cb_facecolor received facecolor args {0}'.format(args))
        self.facecolor = args
        self.queue.put((path, args))
        
    def cb_load(self, path, args):
        print('cb_load received args {0}'.format(args))
        self.load = args
        self.queue.put((path, args))
        
    # def add_callback(self, address, types, func):
    #     self.callbacks.append((address, types, func))
    #     self.server.add_method(address, types, func)
        
    # def cb_mfcc(self, path, args):
    #     # i, f = args
    #     # print("received message '%s' with arguments '%d' and '%f'" % (path, i, f))
    #     self.queue.put(args)
    #     # print('received args {0}'.format(args))

    # def cb_beat(self, path, args):
    #     print('got args = {0}'.format(args))
    #     self.queue.put(args)

    def run(self):
        # loop and dispatch messages every 100ms
        while self.isrunning:
            self.server.recv(100)
        print('terminating')

