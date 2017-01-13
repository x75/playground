"""Little demo on how to use information theoretic measures for assessing the relevance of elements of a feature set for a given
classification or regression task"""


# 2016 oswald berthold

# TODO: compute these measures using our own density estimators: gmm, som
# TODO: clean up for pushing: merge with smp.infth, pull im_quadrotor_plot stuff
# TODO: clear data?

import argparse, os, sys, time
import numpy as np
import pylab as pl

from jpype import startJVM, isJVMStarted, getDefaultJVMPath, JPackage, shutdownJVM, JArray, JDouble, attachThreadToJVM

from smp.infth import init_jpype

from im.im_quadrotor_plot import plot_infth_multi_image

# corr as comparison
# element-wise MI
# check entropy in LCP
# more data: aleke, mfcc
# pi, ais, te, cte

class InfthDataSets(object):
    
    def __init__(self):
        self.datasets = [self.get_data_toy_rec2pol, self.get_data_toy_exp]
        # self.datasets = [self.get_data_toy_rec2pol_noise, self.get_data_toy_exp_noise]
        # self.datasets = [self.get_data_ratslam_conv3, self.get_data_ratslam_rsf]
        # self.datasets = [self.get_data_spider_thin_15, self.get_data_spider_thick_15]
        # self.datasets = [self.get_data_ratslam_rsf]
        # self.datasets = [get_data_toy_rec2pol, get_data_toy_exp, get_data_ratslam_conv3, get_data_ratslam_rsf, get_data_mfcc_motors, get_data_wave_motors]
    
    def get_data_toy_rec2pol(self, numsteps = 1000):
        """create rec2pol data"""
        Y = np.linspace(0, 2*np.pi, numsteps).reshape((numsteps, 1))
        # print "aaahahah", np.cos(Y).shape
        X = np.hstack((np.cos(Y), np.sin(Y)))
        print "rec2pol: X.shape", X.shape, "Y.shape", Y.shape
        return {"X": X, "Y": Y}

    def get_data_toy_rec2pol_noise(self, numsteps = 1000, noise = 1.0):
        """create rec2pol data noisy"""
        data = self.get_data_toy_rec2pol(numsteps = numsteps)
        data["X"][:,0] += np.random.normal(0.0, noise, data["X"][:,0].shape)
        return data
    
    def get_data_toy_exp(self, numsteps = 1000):
        Y = np.linspace(0, 3, numsteps).reshape((numsteps, 1))
        X = np.hstack((Y * np.random.uniform(1.0, 2.0), np.exp(Y), Y**2))
        return {"X": X, "Y": Y}

    def get_data_toy_exp_noise(self, numsteps = 1000, noise = 0.1):
        data = self.get_data_toy_exp(numsteps = numsteps)
        data["X"][:,0] += np.random.normal(0.0, noise, data["X"][:,0].shape)
        return data
    
    def get_data_ratslam_conv3(self, numsteps = 1000):
        X = np.load("conv3.npy").astype(np.float64)
        X /= np.max(np.abs(X))
        print "have nans?", np.any(np.isnan(X)), "isfinite", np.any(np.isfinite(X))
        random_projection = np.random.randint(0, X.shape[1], size = 120)
        # random_projection = np.arange(0, 1000) + 10000
        # print "random_projection", random_projection
        X = X[:,random_projection]
        # X = X[:,[3,4,5]]
        print "have nans?", np.any(np.isnan(X)), "isfinite", np.any(np.isfinite(X))
        print "ratslam conv3.shape", X.shape
        Y = np.load("conv3.pkl")[:-1].astype(np.float64)
        Y /= np.max(np.abs(Y))
        print "ratslam position.shape", Y.shape
        return {"X": X, "Y": Y}

    def get_data_ratslam_rsf(self, numsteps = 1000):
        # X = np.load("conv3.npy").astype(np.float64)
        import cPickle
        X = np.array(cPickle.load(open("daten_fuer_oswald/rsf_yred_4_xred_1.pickle", "rb")))
        X /= np.max(np.abs(X))
        print "have nans?", np.any(np.isnan(X)), "isfinite", np.any(np.isfinite(X))
        random_projection = np.random.randint(0, X.shape[1], size = 40)
        # random_projection = np.arange(0, 1000) + 10000
        # print "random_projection", random_projection
        X = X[:,random_projection]
        # X = X[:,[3,4,5]]
        print "have nans?", np.any(np.isnan(X)), "isfinite", np.any(np.isfinite(X))
        print "ratslam conv3.shape", X.shape
        Y = np.load("conv3.pkl")[:-1].astype(np.float64)
        Y /= np.max(np.abs(Y))
        print "ratslam position.shape", Y.shape
        return {"X": X, "Y": Y}


    def get_data_spider_thin_15(self, numsteps = 1000):
        datafile = "forOswald/no_copy_15retsize_Windpark_Oro_Castle_Goatpeaks_rotated_and_flipped.pkl"
        datafile = "forOswald/no_copy_50retsize_Windpark_Oro_Castle_Goatpeaks_rotated_and_flipped.pkl"
        # datafile = "forOswald/no_copy_117retsize_Windpark_Oro_Castle_Goatpeaks_rotated_and_flipped.pkl"
        import pickle
        data = pickle.load(open(datafile, "rb"))
        X = np.asarray(data["data"]).astype(np.float64)
        Y = np.asarray(data["labels"]).reshape((-1, 1)).astype(np.float64)
        print X.shape, Y.shape
        return {"X": X, "Y": Y}

    def get_data_spider_thick_15(self, numsteps = 1000):
        datafile = "forOswald/no_copy_spider_eye6.86_7.2_15retsize_Windpark_Oro_Castle_Goatpeaks_rotated_and_flipped.pkl"
        datafile = "forOswald/no_copy_spider_eye6.86_7.2_50retsize_Windpark_Oro_Castle_Goatpeaks_rotated_and_flipped.pkl"
        # datafile = "forOswald/no_copy_spider_eye6.86_7.2_117retsize_Windpark_Oro_Castle_Goatpeaks_rotated_and_flipped.pkl"
        import pickle
        data = pickle.load(open(datafile, "rb"))
        X = np.asarray(data["data"]).astype(np.float64)
        Y = np.asarray(data["labels"]).reshape((-1, 1)).astype(np.float64)
        print X.shape, Y.shape
        return {"X": X, "Y": Y}
        
    def get_data_mfcc_motors(self, numsteps = 1000):
        pass

    
    def get_data_wave_motors(self, numsteps = 1000):
        
        pass
    

# for all methods
# argument: data
# argument: gaussian, kernel, kozachenko, kraskov1,2

class InfthMeasures(object):
    def __init__(self):
        pass

    def prepare_data_and_attributes(self, data, check_shape = False): # False
        # prepare data and attributes
        src = np.atleast_2d(data["X"])
        dst = np.atleast_2d(data["Y"])
        # check orientation
        if check_shape:
            if src.shape[0] < src.shape[1]:
                src = src.T
            if dst.shape[0] < dst.shape[1]:
                dst = dst.T
        return src, dst

    def infth_ent_multivariate(self, data, estimator = "kraskov1"):
        """compute multivariate entropy, aka joint entropy for all variables"""
        # self.entmvCalcClass = JPackage("infodynamics.measures.continuous.kozachenko").EntropyCalculatorMultiVariateKozachenko
        self.entmvCalcClass = JPackage("infodynamics.measures.continuous.kernel").EntropyCalculatorMultiVariateKernel
        self.entmvCalc      = self.entmvCalcClass()
        # prepare data and attributes
        src, dst = self.prepare_data_and_attributes(data)
        print "entmv shapes", src.shape, dst.shape
        print "entmv dtypes", src.dtype, dst.dtype
        X = np.hstack((src, dst))
        dim_X = X.shape[1]

        self.entmvCalc.initialise(dim_X)
        self.entmvCalc.setObservations(X)

        entmv_avg = self.entmvCalc.computeAverageLocalOfObservations()
        return entmv_avg

    def infth_ent_sum_single_entropies(self, data):
        """compute the sum of single entropies"""
        self.entCalcClass = JPackage("infodynamics.measures.continuous.kernel").EntropyCalculatorKernel
        self.entCalc      = self.entCalcClass()
        # prepare data and attributes
        src, dst = self.prepare_data_and_attributes(data)
        X = np.hstack((src, dst))
        dim_X = X.shape[1]

        ent_single = []
        ent_avg = 0
        for d in range(dim_X):
            self.entCalc.initialise()
            self.entCalc.setObservations(X[:,d])
            ent_single.append(self.entCalc.computeAverageLocalOfObservations())
        ent_single = np.array(ent_single)
        ent_avg = np.sum(ent_single)
        return ent_avg, ent_single
        
    def infth_mi_multivariate(self, data, estimator = "kraskov1", normalize = True):
        """compute MI multivariate"""
        # init class and instance
        # self.mimvCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
        self.mimvCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2        
        # self.mimvCalcClass = JPackage("infodynamics.measures.continuous.kernel").MutualInfoCalculatorMultiVariateKernel
        self.mimvCalc      = self.mimvCalcClass()
        # set properties
        self.mimvCalc.setProperty("NORMALISE", "true")
        # self.mimvCalc.setProperty("PROP_TIME_DIFF", 0)

        # prepare data and attributes
        src, dst = self.prepare_data_and_attributes(data)
        # src_ = src.copy()
        # src = dst.copy()

        # pl.hist(src[0], bins=255)
        # pl.show()
        
        
        print "mimv shapes", src.shape, dst.shape
        print "mimv dtypes", src.dtype, dst.dtype
        dim_src, dim_dst = src.shape[1], dst.shape[1]
        
        # compute stuff
        # self.mimvCalc.initialise()
        self.mimvCalc.initialise(dim_src, dim_dst)
        self.mimvCalc.setObservations(src, dst)
        # the average global MI between all source channels and all destination channels
        mimv_avg = self.mimvCalc.computeAverageLocalOfObservations()
        return mimv_avg

    def infth_mi_elementwise(self, data):
        """elementwise MI matrix, taken from im/im_quadrotor_plot.py:compute_mutual_information"""
        self.miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
        # miCalcClassC = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov2
        self.miCalcC = self.miCalcClassC()
        self.miCalcC.setProperty("NORMALISE", "true")
        self.miCalcC.setProperty(self.miCalcC.PROP_TIME_DIFF, "0")

        # prepare data and attributes
        src, dst = self.prepare_data_and_attributes(data)
        dim_src, dim_dst = src.shape[1], dst.shape[1]
        
        dim_src, dim_dst = (src.shape[1], dst.shape[1])
        
        measmat  = np.zeros((dim_dst, dim_src))
        
        for m in range(dim_dst):
            for s in range(dim_src):
                # print("m,s", m, s)

                # print("ha", m, motor[:,[m]])
                self.miCalcC.initialise() # sensor.shape[1], motor.shape[1])
                # miCalcC.setObservations(src[:,s], dst[:,m])
                self.miCalcC.setObservations(src[:,[s]], dst[:,[m]])
                mi = self.miCalcC.computeAverageLocalOfObservations()
                # print("mi", mi)
                measmat[m,s] = mi

        return measmat

    def infth_multii_int(self, data):
        """compute Multi-Information / Integration: the difference between sum of
        individual entropies and joint entropy, Tononi, Sporns & Edelman et al. 1994"""
        
        # init class and instance
        self.multiiCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MultiInfoCalculatorKraskov1
        # self.multiiCalcClass = JPackage("infodynamics.measures.continuous.kernel").MultiInfoCalculatorKernel
        self.multiiCalc      = self.multiiCalcClass()
        # set properties
        self.multiiCalc.setProperty("NORMALISE", "true")
        # self.multiiCalc.setProperty("PROP_ADD_NOISE", "true")
        self.multiiCalc.setProperty("SAMPLING_FACTOR_PROP_NAME", "1.0") # how much data to use for estimation
        
        # prepare data and attributes
        src, dst = self.prepare_data_and_attributes(data)
        X = np.hstack((src, dst))
        dim_X = X.shape[1]
        
        # compute stuff
        # self.mimvCalc.initialise()
        self.multiiCalc.initialise(dim_X)
        self.multiiCalc.setObservations(X)
        # the average global MI between all source channels and all destination channels
        multii_avg = self.multiiCalc.computeAverageLocalOfObservations()
        return multii_avg

    def infth_pi(self, data):
        """compute PI"""
        self.piCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
        return 0

    def infth_ais(self, data):
        """compute AIS"""
        self.piCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
        return 0

    def infth_te(self, data):
        """compute TE"""
        self.piCalcClass = JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov1
        return 0

    def infth_corr(self, data):
        src, dst = self.prepare_data_and_attributes(data)
        X = np.hstack((src, dst))
        dim_X = X.shape[1]
        corrcoefs = np.corrcoef(X.T)
        print "corrcoefs nan/finite", np.any(np.isnan(corrcoefs)), np.any(np.isfinite(corrcoefs))
        return corrcoefs

    def infth_lyapunov(self, data):
        pass

    def infth_linear_probe(self, data):
        """learn classifier / regressor probe"""
        from sklearn import linear_model
        import sklearn
        from sklearn import kernel_ridge
        
        lm = linear_model.Ridge(alpha = 0.0)


        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(data["X"], data["Y"], random_state=1)

        # pl.subplot(211)
        # pl.plot(data["Y"])
        # pl.subplot(212)
        # pl.plot(range(y_train.shape[0]), y_train)
        # pl.plot(range(y_train.shape[0], y_train.shape[0]+ y_test.shape[0]), y_test)
        # pl.show()
                        
        # lm.fit(data["X"], data["Y"])
        # Y_ = lm.predict(data["X"]) # training error
        # mse = np.mean(np.square(data["Y"] - Y_))
        lm.fit(X_train, y_train)
        y_ = lm.predict(X_test)
        mse = np.mean(np.square(y_test - y_))

        # print "regression training MSE = %f" % (mse)

        # pl.plot(data["Y"])
        # pl.plot(Y_)
        idx = np.argsort(y_test, axis=0)
        print y_test.shape, idx.shape
        print idx

        y_sorted = y_[idx.flatten()]
        lm2 = linear_model.Ridge(alpha=0.0)
        y_sorted_flat = y_sorted.reshape((-1, 1))
        idx_flat = np.arange(y_sorted.shape[0]).reshape((-1, 1))
        print "shapes", y_sorted_flat.shape, idx_flat.shape
        lm2.fit(idx_flat, y_sorted_flat)
        # print dir(lm2)
        print lm2.coef_, lm2.intercept_

        krr = kernel_ridge.KernelRidge(alpha = 0.0, gamma = 0.01, kernel="rbf")
        krr.fit(X_train, y_train)
        y_krr = krr.predict(X_test)
        y_krr_sorted = y_krr[idx.flatten()]
                
        pl.plot(y_test[idx.flatten()])
        pl.plot(y_sorted_flat)
        pl.plot(y_krr_sorted)
        pl.plot(idx_flat, lm2.coef_ * idx_flat + lm2.intercept_)
        pl.show()
        
        return mse
        
    def infth_learn_tapping(self, data):
        """learn tapping"""
        pass
        
def main():
    doplot = False
    
    init_jpype()

    ids = InfthDataSets()
    ims = InfthMeasures()

    mse_s = []
    
    for i, dataset in enumerate(ids.datasets):
        # print dataset
        data = dataset(numsteps = 1000)
        print "data shapes", data["X"].shape, data["Y"].shape
        # print data
        # X,Y = data["X"], data["Y"]
        X, Y = ims.prepare_data_and_attributes(data, check_shape = False)
        
        entmv = ims.infth_ent_multivariate(data)
        print "Joint Entropy H(X) = %f nats" % (entmv)
        # ent_sum, ent_single = ims.infth_ent_sum_single_entropies(data)
        # print "Sum Single Entropies H(X) = %f nats" % (ent_sum)
        # print "Single Entropies H(X_i) = %s nats" % (ent_single)
        # mimv = ims.infth_mi_multivariate(data)
        # print "Global Mutual Information MI(src; dst) = %f nats (%s)" % (mimv, dataset)
        # multii = ims.infth_multii_int(data)
        # print "Multi-Information / Integration I(X) = %f nats" % (multii)
        # print "Multi-Information / Integration I(X) = %f nats (using definition by Tononi 1994)" % (ent_sum - entmv)
        # corrcoefs = ims.infth_corr(data)
        # # print "Correlation coefficients = %s" % (str(corrcoefs))
        # print "Correlation coefficients, min = %f, max = %f" % (np.min(corrcoefs), np.max(corrcoefs))
        # mimat = ims.infth_mi_elementwise(data)
        # print "Mutual Information element-wise min = %f, max = %f" % (np.min(mimat), np.max(mimat))
        # # TODO: compute historgram over flattened upper triangular mimat
        # probe_reg_mse = ims.infth_linear_probe(data)
        # print "Linear Probe MSE = %f" % (probe_reg_mse)


        # mse_s.append(probe_reg_mse)
        
        ################################################################################
        # plotting
        if doplot:
            fig = pl.figure() # figsize=(figw, figh)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            
            ax1.plot(X)
            ax1.plot(Y)

            dimlim = 10
            if mimat.shape[0] > dimlim or mimat.shape[1] > dimlim:
                print "plotting histograms instead of full matrix"
                mimat_hist = np.histogram(mimat.flatten(), bins=100)
                ax2.bar(mimat_hist[1][:-1], mimat_hist[0], mimat_hist[1][1] - mimat_hist[1][0])
                mithresh = 0.2
                print "# of features with element-wise MI > %f = %d/%d" % (mithresh, mimat[mimat > mithresh].flatten().shape[0], data["X"].shape[1] * data["Y"].shape[1])

                if not np.any(np.isfinite(corrcoefs)):
                    idx = np.triu_indices(corrcoefs.shape[0], k = 0)
                    print "idx", idx, corrcoefs.shape
                    corrcoefs_hist = np.histogram(corrcoefs[idx], bins=100)
                    ax3.bar(corrcoefs_hist[1][:-1], corrcoefs_hist[0], corrcoefs_hist[1][1] - corrcoefs_hist[1][0])
            else:
                plot_infth_multi_image(i, fig, ax2, mimat, "MI matrix")
                plot_infth_multi_image(i, fig, ax3, corrcoefs, "corrcoef matrix")
        
            pl.show()

    return mse_s

def main_dimstack():
    from smp.dimstack import dimensional_stacking

    # srcdim = 4
    # edgelen = 10
    # dimdata = np.zeros([edgelen] * srcdim)
    # print "dimdata.shape", dimdata.shape
    # # for dim in range(4):
    # #     dimdata.append(np.random.uniform(0, dim+1, (20**2,)))
    # # dimdata = np.random.uniform(0, 1, (10, 10, 10, 10))
    # dimdata = np.random.uniform(0, np.array([i for i in range(10)]),
    #                             (10, 10, 10, ))


    # print "dimdata[0,0,0]", dimdata[0,0,1]
    # doesnt work like this, use meshgrid 4D
        
    # dimdata = np.array(dimdata)

    edgelen = 10
    d0 = np.linspace(0, 0.25*np.pi, edgelen)
    d1 = np.linspace(0, 0.5*np.pi, edgelen)
    d2 = np.linspace(0, 0.75*np.pi, edgelen)
    # d3 = np.linspace(0, 1.0*np.pi, edgelen)
    d3 = np.zeros((edgelen,))
    eps = 1e-8

    d0_, d1_, d2_, d3_ = np.meshgrid(d0, d1, d2, d3)

    a1 = d0_**2 + d1_**2 + d2_**2 + d3_**2
    a2 = d0_**2 + d1_**2 + d2_**2 # + d3_**2
    # z = np.sin(a1) / (np.sin(-a2 + np.random.uniform(0, np.pi)) + eps)**(1/2)
    z = np.sin(a1)
    # print z.shape
    # h = pl.contourf(d0,d1,z)
    # pl.show()
    
    print "z.shape", z.shape #, z
    
    xdims = [0, 2]
    ydims = [1, 3]
            
    stacked_data = dimensional_stacking(z, xdims, ydims)

    print "stacked_data.shape", stacked_data.shape
    # print "stacked_data", stacked_data

    # pl.imshow
    pl.subplot(121)
    pl.pcolormesh(stacked_data)
    pl.gca().set_aspect(1.0)

    # t = np.linspace(0, 1, 1000)
    t = np.random.uniform(-np.pi, np.pi, 10000)

    t1 = t * 1e-3
    # X = np.array([np.cos(t*1e-1)**2, np.cos(t * 0.0013), np.cos(t*0.001)/(t1+1e-9), np.sin(t*1.0)/(t1+1e-9)]).T
    # X = np.array([np.sin(t1**2), np.exp(-(t**2)), t**3, t**0.5]).T
    X = np.array([t*0.1, t*0, t*0, t*0]).T # t*0.12, t*0.14, t*0.16]).T
    Xn = np.random.normal([0.1, 0.5, 1.0, 2.0], [0.2, 1.0, 0.5, 0.1], (10000, 4)) #, X.shape)
    X = X + Xn
    print "X.shape", X.shape
    # use a histogram, that is scatterstack
    Xh = np.histogramdd(X, bins=10)
    # print "type(Xh)", type(Xh)
    print "histo shape Xh", Xh[0].shape
    # Xh += Xn.reshape((10, 10, 10, 10))

    Xh[0][0,:,:,:] = 20.0
    Xh[0][1,:,:,:] = 30.0
    Xh[0][:,1,:,:] = 40.0
    Xh[0][:,:,2,:] = 50.0
    Xh[0][:,:,4,:] = 70.0
    Xh[0][:,:,:,7] = 100.0
    
    stacked_data = dimensional_stacking(Xh[0], xdims, ydims)

    print "stacked_data.shape", stacked_data.shape
    # , stacked_data

        
    pl.subplot(122)
    x_ = np.linspace(-1.0, 1.0, 10**2)
    y_ = np.linspace(-1.0, 1.0, 10**2)
    pl.pcolormesh(x_, y_, stacked_data)
    pl.gca().set_aspect(1.0)
    pl.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="infth", type=str, help="which mode to run: infth, dimstack")
    args = parser.parse_args()
    
    # definitions of entropy, joint entropy, conditional entropy, etc from Lizier 2014 JIDT Paper/Cheatsheet

    if args.mode == "infth":    
        mse_s = []
        for i in range(10):
            # mse_s = main()
            mse_s.append(main())
        print "mse_s", mse_s
        mse_s = np.array(mse_s)
        print "conv3 avg mse over 100 runs @120-dim proj = %f" % mse_s[:,0].mean() # mse_s[range(0, 10, 2)]
        print "  rsf avg mse over 100 runs @040-dim proj = %f" % mse_s[:,1].mean() # mse_s[range(1, 10, 2)]
    elif args.mode == "dimstack":
        main_dimstack()
