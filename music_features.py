
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import pickle

import essentia as e
import essentia.standard as estd

# from essentia.standard import *
# import essentia.streaming

# print dir(essentia.standard)

def makefig(rows = 1, cols = 1):
    fig = plt.figure()
    gs = GridSpec(rows, cols)
    for i in gs:
        fig.add_subplot(i)
    return fig

def loadaudio(args):
    loader = estd.MonoLoader(filename= args.file, sampleRate = args.samplerate)
    return loader()

def main_mfcc(args):

    plt.ion()

    audio = loadaudio(args)
    
    print "audio", type(audio), audio.shape

    # pylab contains the plot() function, as well as figure, etc... (same names as Matlab)
    plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

    fig = makefig(rows = 2, cols = 2)

    w = estd.Windowing(type = 'hann')
    spectrum = estd.Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = estd.MFCC()

    print "w", repr(w)
    print "spectrum", repr(spectrum)
    print "mfcc", repr(mfcc)

    frame = audio[int(0.2*args.samplerate) : int(0.2*args.samplerate) + 1024]
    print "frame.shape", frame.shape
    spec = spectrum(w(frame))
    mfcc_bands, mfcc_coeffs = mfcc(spec)


    print "type(spec)", type(spec)
    print "spec.shape", spec.shape

    fig.axes[0].plot(audio[int(0.2*args.samplerate):int(0.4*args.samplerate)])
    fig.axes[0].set_title("This is how the 2nd second of this audio looks like:")
    # plt.show() # unnecessary if you started "ipython --pylab"

    fig.axes[1].plot(spec)
    fig.axes[1].set_title("The spectrum of a frame:")

    fig.axes[2].plot(mfcc_bands)
    fig.axes[2].set_title("Mel band spectral energies of a frame:")

    fig.axes[3].plot(mfcc_coeffs)
    fig.axes[3].set_title("First 13 MFCCs of a frame:")

    fig.show()

    # plt.show() # unnecessary if you started "ipython --pylab"
    ################################################################################
    fig2 = makefig(rows = 2, cols = 2)


    mfccs = []
    melbands = []

    for frame in estd.FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    mfccs = np.array(mfccs).T
    melbands = np.array(melbands).T


    pool = e.Pool()

    for frame in estd.FrameGenerator(audio, frameSize = 1024, hopSize = 512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        pool.add('lowlevel.mfcc', mfcc_coeffs)
        pool.add('lowlevel.mfcc_bands', mfcc_bands)

    fig2.axes[2].imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', origin='lower', interpolation='none')
    fig2.axes[2].set_title("Mel band spectral energies in frames")

    fig2.axes[3].imshow(pool['lowlevel.mfcc'].T[1:,:], aspect='auto', origin='lower', interpolation='none')
    fig2.axes[3].set_title("MFCCs in frames")



    # and plot
    fig2.axes[0].imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
    fig2.axes[0].set_title("Mel band spectral energies in frames")
    # show() # unnecessary if you started "ipython --pylab"

    fig2.axes[1].imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
    fig2.axes[1].set_title("MFCCs in frames")

    fig2.show()


    plt.ioff()
    plt.show() # unnecessary if you started "ipython --pylab"

def main_danceability(args):
    audio = loadaudio(args)
    
    # create the pool and the necessary algorithms
    pool = e.Pool()
    w = estd.Windowing()
    spec = estd.Spectrum()
    centroid = estd.SpectralCentroidTime()

    # compute the centroid for all frames in our audio and add it to the pool
    for frame in estd.FrameGenerator(audio, frameSize = 1024, hopSize = 512):
        c = centroid(spec(w(frame)))
        pool.add('lowlevel.centroid', c)

    # aggregate the results
    aggrpool = estd.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)


    # create the pool and the necessary algorithms
    pool = e.Pool()
    w = estd.Windowing()
    # spec = estd.Spectrum()
    # centroid = estd.SpectralCentroidTime()
    danceability = estd.Danceability(maxTau = 10000, minTau = 300, sampleRate = args.samplerate)
    
    # compute the centroid for all frames in our audio and add it to the pool
    for frame in estd.FrameGenerator(audio, frameSize = 10 * args.samplerate, hopSize = 5 * args.samplerate):
        dreal, ddfa = danceability(w(frame))
        print "d", dreal # , "frame", frame
        pool.add('rhythm.danceability', dreal)

    print type(pool['rhythm.danceability'])
        
    # aggregate the results
    # aggrpool = estd.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)
    
    # write result to file
    # estd.YamlOutput(filename = args.file + '.features.yaml')(aggrpool)

    fig = makefig(rows = 2, cols = 2)
    ax = fig.axes

    ax[0].plot(pool['rhythm.danceability'])

    plt.show()

def main_extractor(args):
    audio = loadaudio(args)
    
    frame = audio

    extr = estd.Extractor()

    # # compute the centroid for all frames in our audio and add it to the pool
    # for frame in estd.FrameGenerator(audio, frameSize = 10 * args.samplerate, hopSize = 5 * args.samplerate):
    #     dreal, ddfa = danceability(w(frame))
    #     print "d", dreal # , "frame", frame
    #     pool.add('rhythm.danceability', dreal)
    
    p = extr(frame)

    pdict = {}
    # print "type(p)", type(p)
    for desc in p.descriptorNames():
        print desc, type(p[desc])
        #     print "{0: >20}: {1}".format(desc, p[desc])
        pdict[desc] = p[desc]

    pickle.dump(pdict, open("data/music_features_%s.pkl" % (args.file.replace("/", "_"), ), "wb"))

def main_extractor_pickle_plot(args):
    from matplotlib import rcParams
    rcParams['axes.titlesize'] = 7
    rcParams['axes.labelsize'] = 6
    rcParams['xtick.labelsize'] = 6
    rcParams['ytick.labelsize'] = 6
    # load bag of features computed above from a pickle
    pdict = pickle.load(open(args.file, "rb"))

    # get sorted keys
    feature_keys = sorted(pdict.keys())

    feature_groups = np.unique([k.split(".")[0] for k in feature_keys])

    # print np.unique(feature_keys_groups)

    feature_keys_groups = {}
    for group in feature_groups:
        feature_keys_groups[group] = [k for k in feature_keys if k.split(".")[0] == group]

    print len(feature_keys_groups)

    for group in feature_groups:
        fk_ = feature_keys_groups[group]
        plot_features(pdict, feature_keys = fk_, group = group)
    
    plt.show()


def plot_features(pdict, feature_keys, group):
    numrows = int(np.sqrt(len(feature_keys)))
    numcols = int(np.ceil(len(feature_keys)/float(numrows)))

    # print numrows, numcols
    
    fig = plt.figure()
    fig.suptitle(
        "essentia extractor %s features for %s" % (group, args.file, ),
        fontsize = 8)
    gs = GridSpec(numrows, numcols, hspace = 0.2)

    for i, k in enumerate(feature_keys):
        ax = fig.add_subplot(gs[i])
        if type(pdict[k]) is np.ndarray:
            ax.plot(pdict[k], alpha = 0.5, linewidth = 0.8, linestyle = "-")
            title = "%s/%s" % (k.split(".")[1], pdict[k].shape)
            if i/numcols < (numrows - 1):
                ax.set_xticklabels([])
        elif type(pdict[k]) is float:
            title = "%s/%s" % (k.split(".")[1], type(pdict[k]))
            ax.text(0, 0, "%f" % (pdict[k],), fontsize = 8)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
        elif type(pdict[k]) is str:
            title = "%s/%s" % (k.split(".")[1], type(pdict[k]))
            ax.text(0, 0, "%s" % (pdict[k],), fontsize = 8)
            ax.set_xlim((-1, 1))
            ax.set_ylim((-1, 1))
            
        ax.set_title(title)

    fig.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Input file [data/ep1.wav]", type = str, default = "data/ep1.wav")
    parser.add_argument("-m", "--mode", help="Program mode [mfcc]: mfcc, danceability, extractor, extractor_plot", type = str, default = "mfcc")
    parser.add_argument("-sr", "--samplerate", help="Sample rate to use [44100]", type = int, default = 44100)

    args = parser.parse_args()

    if args.mode == "mfcc":
        main_mfcc(args)
    elif args.mode == "danceability":
        main_danceability(args)
    elif args.mode == "extractor":
        main_extractor(args)
    elif args.mode == "extractor_plot":
        main_extractor_pickle_plot(args)
    
