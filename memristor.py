#!/usr/bin/env python
import sys, csv
import numpy as np
#import matplotlib.pyplot as plt
import pylab as pl
#from optparse import OptionParser

def f (V ):
    alpha = 0.5
    beta = 4
    VT = 1.0
    return -(beta * V) + (0.5 * (beta - alpha) * \
                          (np.abs(V + VT) - np.abs(V - VT)))

def main(argv):
    t = np.linspace(0, 1, 100)
    V = (t * 4.0) - 2
    pl.plot(V,f(V))
    pl.axhline(0, c="black")
    pl.axvline(0, c="black")
    pl.show()

    np.savetxt("ana-memristor-funktion.dat", np.vstack((V, f(V))).T)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
