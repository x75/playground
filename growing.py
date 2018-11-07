"""Idea: self-organization, two pathways:
 - horizontal
 - vertical (stacked, deep, onion, ...)

Moved to and developed in smp_growth project
"""

import time, sys, argparse
import numpy as np
import matplotlib.pyplot as plt

types = ["RBF", "tanh", "local_linear_reg_rls", "randomBF", "res", "kerasmodel"]

class unitID(object):
    def __init__(self, ndim_in = 1):
        self.ndim_in = ndim_in
        self.a = np.zeros((1, self.ndim_in))
        
    def activate(self, x):
        self.a = x
        return self.a

class unitRBF(object):
    def __init__(self, ndim_in = 1):
        self.ndim_in = ndim_in
        self.eta1 = 0.01
        # self.w = np.zeros((ndim_in, 1))
        self.w = np.random.uniform(-1e-1, 1e-1, size=(self.ndim_in, 1))
        self.a = np.zeros((1,self.ndim_in))

    def activate(self, x1, x2):
        print "x1.shape = %s, x2.shape = %s" % (x1.shape, x2.shape)
        # x = np.vstack((x1.T, x2.T))
        self.a = self.w - x1.T
        # self.a = # np.abs(diff)
        self.a += x2.T
        self.dw = self.a * self.eta1
        self.w -= self.dw
        print "self.a.shape", self.a.shape, self.w.shape, self.dw.shape, np.linalg.norm(self.dw)
        return self.a

class Nv1(object):
    def __init__(self, maxsize = 10, unittype = "RBF", ndim_in = 1):
        self.maxsize = maxsize
        self.ndim_in = ndim_in
        self.layers = [unitID] + [unitRBF(ndim_in = self.ndim_in)] * 2 + [None for i in range(2, self.maxsize)]
        # layer activations: input layer + number of layers (maxsize) + self.activation
        self.al = np.zeros((self.maxsize + 2, ndim_in * 2))
        self.a  = np.zeros((1,1))

    def activate(self, x):
        print "x.shape", x.shape, self.layers[1].w.shape
        self.al[[0],:self.ndim_in] = x.copy()
        for li in range(1, len(self.layers)):
            print "li = %d, l = %s" % (li, self.al[li])
            if self.al[li] is not None:
                self.al[[li]] = self.layers[li].activate(self.al[[li-1]], self.al[[li+1]])

class Nv2(object):
    def __init__(self, maxsize = 10, unittype = "RBF", ndim_in = 1):
        self.maxsize = maxsize
        self.ndim_in = ndim_in
        self.layers = [unitID, unitRBF(ndim_in = self.ndim_in)] + [None for i in range(1, self.maxsize)]
        # layer activations: input layer + number of layers (maxsize) + self.activation
        self.al = np.zeros((self.maxsize + 2, ndim_in))
        self.a  = np.zeros((1,1))
        self.r  = np.zeros((self.maxsize + 2, ))
        self.r_ = np.zeros_like(self.r)

    def activate(self, x):
        print "x.shape", x.shape, self.layers[1].w.shape
        self.al[[0]] = x.copy()
        # forward
        for li in range(1, len(self.layers)):
            print "li = %d, l = %s" % (li, self.al[li])
            if self.layers[li] is not None:
                if self.layers[li+1] is not None:
                    x2 = self.al[[li+1]]
                else:
                    x2 = np.zeros((self.al[[li]].shape))
                self.al[[li]] = self.layers[li].activate(self.al[[li-1]], x2).T
                
        # backward
        for li in range(len(self.layers) - 1, 1, -1):
            print "li = %d, l = %s" % (li, self.al[li])
            if self.layers[li] is not None:
                if self.layers[li+1] is not None:
                    x2 = self.al[[li+1]]
                else:
                    x2 = np.zeros((self.al[[li]].shape))
                self.al[[li]] = self.layers[li].activate(self.al[[li-1]], x2).T
        self.r = np.sum(np.abs(self.al), axis=1)
        print "self.r = %s, self.w" % (self.r)
        self.r_ = 0.99 * self.r_ + 0.01 * self.r

class Nv3(object):
    def __init__(self, maxsize = 10):
        self.maxsize = 10
                
def generate_data_1(numsteps):
    from pypr.clustering import *
    # from numpy import *
    centroids=[np.array([0,-3]), np.array([1.5, -1])]
    ccov = [np.array([[0.1,-0.3],[0.3,0.12]]), np.diag([0.1, 0.4])]
    mc   = [0.5, 0.5]
    samples = numsteps
    return gmm.sample_gaussian_mixture(centroids = centroids, ccov = ccov, mc = mc, samples=samples)    
        
def train_network(x):

    # L1 activate
    # for L in L-exhausted
    #    activate L on x
    #    if capacity_L > theta:
    #        train L(x)
    #    track residuals r_L
    # if accum residual L_{-1} > theta
    #    spawn new layer

    # aspects:
    # - backward skip connections: upper L activation propagates down to lower L input (recurrence, how to handle that)
    # - activation cycle: activate all the way to the top, back-propagate all the way back down
    # - regain capacity: if residual gets low enough, can learn again

    # v1: learn L1, fixate forever, learn L2, ..., accumulate forward skip conns (inputs getting wider, put PCA/ICA/SFA on connections)
    # v2: learn L1 ...

    # units: RBF, local linear reg (RLS), randomBF, reservoir, full-scale deep network
    pass
    
def main(args):
    print "args", args
    numsteps = 1500
    maxsize = 3

    # generate network
    # net = Nv1(maxsize = 10, unittype = "RBF", ndim_in = 2)
    net = Nv2(maxsize = maxsize, unittype = "RBF", ndim_in = 2)

    print "net", net    
    # generate data
    d = generate_data_1(numsteps)
    print "d.shape", d.shape
    # plt.plot(d[:,0], d[:,1], "ro", alpha=0.5)
    plt.subplot(211)
    plt.plot(d)
    plt.subplot(212)
    plt.hist2d(d[:,0], d[:,1], bins=16)
    plt.colorbar()
    plt.show()

    # log data
    net_r  = np.zeros((numsteps, maxsize+2))
    net_r_ = np.zeros((numsteps, maxsize+2))
    
    # loop over data
    for i in range(numsteps):
        net.activate(d[[i]])
        net_r[i]  = net.r
        net_r_[i] = net.r_

    plt.subplot(211)
    plt.plot(net_r)
    plt.subplot(212)
    plt.plot(net_r_)
    plt.show()

    # train network

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    
    main(args)
