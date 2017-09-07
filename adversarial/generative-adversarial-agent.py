"""generative adversarial agent (gaa)

quick sketch
"""

import argparse
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from smp_base.models_actinf import smpGMM, smpIGMM, smpKNN
from smp_base.learners import smpSHL

def noisef(x):
    return np.random.normal(0, 1.0, size = x.shape) * 0.001
 
def transf0(x):
    # return
    x_ = x + noisef(x)
    # return np.tanh(2.0 * x_)
    return np.clip(x_, -1.1, 1.1)

def transf1(x):
    return (0.7 * x) + 0.1 + noisef(x)

def transf2(x):
    lim = 1.0
    liminv = 1/lim
    x_ = signal.sawtooth(lim * np.pi * x + np.pi * 0.5, width = 0.5) * liminv # * x
    return x_ + noisef(x)

class gaasys(object):
    def __init__(self, args):
        self.dim = args.dim
        self.order = args.order
        self.transf = transf1
        self.x = np.random.uniform(size = (self.dim, 1)) * 0.0
        self.limits = np.ones_like(self.x) * 0.3

    def step(self, x):
        # print "step x = ", x, self.x
        self.x = self.transf(x)
        # print "step x = ", self.x
        return self.x

class mdl1(object):
    def __init__(self, args):
        if hasattr(args, 'idim'):
            self.idim = args.idim
        else:
            self.idim = args.dim
        if hasattr(args, 'odim'):
            self.odim = args.odim
        else:
            self.odim = args.dim
            
        self.modelsize = 100
        self.eta = 1e-3
        self.h = np.random.uniform(-1, 1, size = (self.modelsize, 1))
        self.X_ = np.random.uniform(-1, 1, size = (self.odim, 1))
        self.e_X = np.random.uniform(-1, 1, size = (self.odim, 1))
        self.w_Xh  = np.random.normal(0, 1.0, size = (self.modelsize, self.idim))
        self.w_hX_ = np.random.normal(0, 1.0, size = (self.odim, self.modelsize)) * 1e-6

    def step(self, X, Y):
        # fit
        self.e_X = self.X_ - Y
        # print "e", self.e_X.shape, "h", self.h.shape
        dw = self.eta * np.dot(self.h, -self.e_X).T
        # print "dw", dw.shape, self.w_hX_.shape
        self.w_hX_ += dw
        # predict
        self.h = np.dot(self.w_Xh, X)
        self.X_ = np.dot(self.w_hX_, self.h)
        return self.X_

def main_baseline(args):
    gs = gaasys(args)
    # gs.transf = transf0
    gs.transf = np.random.choice([transf0, transf1, transf2])

    # mdl_cls = smpGMM
    mdl_cls = smpIGMM
    # mdl_cls = smpKNN
    # mdl_cls = smpSHL
    
    # models
    m_g = mdl1(args)
    setattr(args, 'idim', args.dim * 2)
    setattr(args, 'odim', args.dim)
    m_i = mdl1(args)

    # error
    m_e_conf = mdl_cls.defaults
    m_e_conf.update({
        'idim': args.dim, 'odim': args.dim, 'n_neighbors': 8,
        'prior': 'random', 'prior_width': 1.0, 'fit_interval': 100,
        'eta': 1e-3, 'w_input': 10.0, 'w_bias': 1.0, 'modelsize': 200,
        'theta_state': 0.02, 'lrname': 'FORCE', 'spectral_radius': 0.01,
        'tau': 1.0, 'alpha': 10.0, 'wgt_thr': 10, 'mixcomps': 12, 'oversampling': 1,
        'visualize': True, 'input_coupling': 'normal',
        'sigma_mu': 5e-3, 'sigma_sig': 5e-9, 'sigma_pi': 5e-4
        })
    m_e = mdl_cls(conf = m_e_conf)
    # m_e = smpGMM(conf = m_e_conf)
    
    # inverse
    # m_i_conf = smpGMM.defaults
    # m_i_conf.update({'idim': args.dim * 2, 'odim': args.dim, 'em_max_iter': 10})
    # m_i = smpGMM(conf = m_i_conf)

    m_i_conf = mdl_cls.defaults
    m_i_conf.update({
        'idim': args.dim * 2, 'odim': args.dim, 'n_neighbors': 5,
        'prior': 'random', 'prior_width': 0.1, 'fit_interval': 100,
        'eta': 1e-3, 'w_input': 10.0, 'w_bias': 1.0, 'modelsize': 200,
        'theta_state': 0.02, 'lrname': 'FORCE', 'spectral_radius': 0.01,
        'tau': 1.0, 'alpha': 10.0, 'wgt_thr': 10, 'mixcomps': 12, 'oversampling': 1,
        'visualize': True, 'input_coupling': 'normal',
        'sigma_mu': 5e-3, 'sigma_sig': 5e-9, 'sigma_pi': 5e-3
        })
    m_i = mdl_cls(conf = m_i_conf)
    # m_i = smpGMM(conf = m_i_conf)

    X = np.zeros((args.dim, args.numsteps))
    X_ = np.zeros((args.dim, args.numsteps))
    Y = np.zeros((args.dim, args.numsteps))
    Y_ = np.zeros((args.dim, args.numsteps))
    
    e_X_ = np.zeros((args.dim, args.numsteps))
    E_argmin_ = np.zeros((args.dim, args.numsteps))
    E_min_ = np.zeros((args.dim, args.numsteps))
    X_e_min_ = np.zeros((args.dim, args.numsteps))
    E_argmax_ = np.zeros((args.dim, args.numsteps))
    E_max_ = np.zeros((args.dim, args.numsteps))
    X_e_max_ = np.zeros((args.dim, args.numsteps))

    fig = plt.figure()
    fig.show()
    plt.ion()
    gspc = GridSpec(2, 2)
    ax = fig.add_subplot(gspc[0,0])
    ax1 = fig.add_subplot(gspc[0,1])
    ax2 = fig.add_subplot(gspc[1,0])
    ax3 = fig.add_subplot(gspc[1,1])
    ax.set_title('transfer = %s' % (gs.transf))
    t = np.linspace(-1.2, 1.2, 101)
    ax.plot(t, gs.transf(t))
    ax.set_ylim([-1.3, 1.3])
    # plt.draw()
    # plt.pause(1e-9)

    # goal sampler
    # inverse model
    # system
    
    for i in range(1, args.numsteps):
        
        # X_[...,[i]] = m_g.step(X[...,[-1]], x)
        resample_interval = 20
        if i % resample_interval == 0:
            # X_[...,[i]] = np.random.uniform(-1.5, 1.5, size = (args.dim, 1))
            if (i/resample_interval) % 2 == 1:
                print "sampling cooperative"
                mu = X_e_min_[...,[i-1]]
                sig = E_min_[...,[i-1]]
                sig = 0.5
                e_ = np.random.uniform(0, 0.2, size = X[...,[i]].shape)
            else:
                print "sampling adversarial"
                mu = X_e_max_[...,[i-1]]
                sig = E_max_[...,[i-1]]
                sig = 0.1
                e_ = np.random.uniform(0.2, 1.0, size = X[...,[i]].shape)
            sample = m_e.predict(e_) # + noisef(e_) * np.square(e_)
            X_[...,[i]] = sample
            # X_[...,[i]] = np.random.normal(mu, sig)
        else:
            X_[...,[i]] = X_[...,[i-1]].copy()
            
        # print "X_", X_[...,[i]]

        # Y[...,[i]] = np.random.uniform(-1, 1, size = (args.dim, 1))
        # Y[...,[i]] = Y[...,[i-1]] + (np.random.normal(0, 1, size = (args.dim, 1)) * 0.1)
        nu = np.random.normal(0, 1, size = (args.dim, 1)) * 0.02
        # print "gs.x", gs.x, "nu", nu
        # y = gs.x + nu
        
        X_m_i = np.vstack((X_[...,[i]], X[...,[i-1]]))
        # Y_[...,[i]] = np.tanh(1.0 * m_i.predict(X_m_i.T)) + nu
        Y_[...,[i]] = 1.0 * m_i.predict(X_m_i.T) + nu
        # print "Y_", Y_[...,[i]]
        
        # print "y", y
        # Y[...,[i]] = y.copy()
        # print "X", X[...,[i-1]], "Y", Y[...,[i]]
        x = gs.step(Y_[...,[i]])
        # print "x", x
        X[...,[i]] = x.copy()

        e = X_[...,[i]] - X[...,[i]]
        e_X = np.tanh(np.square(e))
        e_X_[...,[i]] = e_X

        # fit error
        m_e.fit(X = e_X.T, y = X[...,[i]].T)#, update = True)

        # print "error sampling"
        E_ = np.random.uniform(0, 1, size = (e_X.shape[0], 10))
        # X_e = np.zeros_like(E_)
        # for i in range(100):
        # print "i", i
        # X_e[...,[i]] = m_e.predict(E_[...,[i]].T).T
        X_e = m_e.predict(E_.T).T
        argmin = np.argmin(E_)
        E_argmin_[...,[i]] = argmin
        # print "argmin", E_argmin_
        E_min_[...,[i]] = E_[...,[argmin]]
        X_e_min_[...,[i]] = X_e[...,[argmin]]
        # print "X_e min", E_argmin_, E_min_, X_e_min_
        argmax = np.argmax(E_)
        E_argmax_[...,[i]] = argmax
        E_max_[...,[i]] = E_[...,[argmax]]
        X_e_max_[...,[i]] = X_e[...,[argmax]]
        # print "X_e max", E_argmax_, E_max_, X_e_max_

        # print "fitting inverse"
        
        # fit inverse
        
        # X_m_i = np.vstack((X[...,1:i], X[...,0:i-1]))
        # Y_m_i = Y_[...,1:i]
        # # print "Xmi", X_m_i
        # # print "Ymi", Y_m_i
        
        # if i > 0 and i % 100 == 0:
        #     m_i.fit(X = X_m_i.T, y = Y_m_i.T)
        
        X_m_i = np.vstack((X[...,[i]], X[...,[i-1]]))
        Y_m_i = Y_[...,[i]]
        # m_i.fit(X = X_m_i.T, y = Y_m_i.T, update = True)
        m_i.fit(X = X_m_i.T, y = Y_m_i.T)

        if i % 100 == 0 or i == (args.numsteps - 1):
            ax1.clear()
            ax1.plot(e_X_.T, '-o', alpha = 0.5, label = 'e_X_')
            
            ax2.clear()
            ax2.plot(X_.T, '-o', alpha = 0.5, label = 'X_')
            ax2.plot(X.T, '-o', alpha = 0.5, label = 'X')
            ax2.plot(Y_.T, '-x', alpha = 0.5, label = 'Y')
            ax2.legend()
            ax3.clear()
            ax3.plot(
                E_min_.T[i-100:i],
                X_e_min_.T[i-100:i], 'o', alpha = 0.5, label = 'X_e_min')
            ax3.plot(
                E_max_.T[i-100:i],
                X_e_max_.T[i-100:i], 'o', alpha = 0.5, label = 'X_e_max')
            ax3.legend()
            plt.draw()
            plt.pause(1e-9)
            
    plt.ioff()
    plt.show()
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # gaa1: 
    parser.add_argument('-m', '--mode', type=str, default='baseline', help = 'Program mode [baseline], one of baseline, gaa1, gaa2, gaa3')
    parser.add_argument('-d', '--dim',  type=int, default=1, help = 'Number of system dimensions [1]')
    parser.add_argument('-n', '--numsteps',  type=int, default=1000, help = 'Number of steps [1000]')
    parser.add_argument('-o', '--order',  type=int, default=0, help = 'System order [0], 0: kinematic, 1: first order, 2: second order')
    parser.add_argument('-s', '--seed',   type=int, default=0, help = 'Random seed [0]')

    args = parser.parse_args()

    np.random.seed(args.seed)
    
    if args.mode == 'baseline':
        main_baseline(args)
