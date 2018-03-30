"""seq2graph

convert sequences into linear trees and accumulate a graph

- single trees
- forests (var len or whatever seq datasets)
- recurrence analysis, modulate node size with occurence/recurrence count (!!!)
- more fancy graph ops

recurrence plots, zigzag, l-systems, growth, scaling, ...

leafs are primitive tokens: words, n-grams, ...
"""
import argparse, signal, sys

import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore, USE_PYSIDE, USE_PYQT5
# from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.widgets.RawImageWidget import RawImageWidget

if USE_PYSIDE:
    import VideoTemplate_pyside as VideoTemplate
elif USE_PYQT5:
    import VideoTemplate_pyqt5 as VideoTemplate
else:
    import VideoTemplate_pyqt as VideoTemplate

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# from https://coldfix.de/2016/11/08/pyqt-boilerplate/#keyboardinterrupt-ctrl-c
# Call this function in your main after creating the QApplication
def setup_interrupt_handling():
    """Setup handling of KeyboardInterrupt (Ctrl-C) for PyQt."""
    signal.signal(signal.SIGINT, _interrupt_handler)
    # Regularly run some (any) python code, so the signal handler gets a
    # chance to be executed:
    safe_timer(50, lambda: None)


# Define this as a global function to make sure it is not garbage
# collected when going out of scope:
def _interrupt_handler(signum, frame):
    """Handle KeyboardInterrupt: quit application."""
    QtGui.QApplication.quit()

def safe_timer(timeout, func, *args, **kwargs):
    """
    Create a timer that is safe against garbage collection and overlapping
    calls. See: http://ralsina.me/weblog/posts/BB974.html
    """
        
    def timer_event():
        try:
            func(*args, **kwargs)
        finally:
            QtCore.QTimer.singleShot(timeout, timer_event)
    QtCore.QTimer.singleShot(timeout, timer_event)

def main_tree(args):
    print('tree ready to roll :)')
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(
        subset='train', categories=categories,
        shuffle=True, random_state=42)

    # inspect twenty_train
    for k in dir(twenty_train):
        t_k = getattr(twenty_train, k)
        t_k_size = 1
        try:
            t_k_size = len(t_k)
        except Exception as err:
            print('%s has no len, see %s' % (k, err))
        print('%s = %s, %s' % (k, type(t_k), t_k_size, ))

    print('-' * 80)

    # data, target, target_name[target]
    for i in range(10):
        print('data[%d] = %s' % (i, len(twenty_train.data[i])))

    # random index
    data_idx = np.random.choice(range(len(twenty_train.data)), (50,)).flatten()
    print('data_idx = %s, %s, %s' % (type(data_idx), data_idx.dtype, data_idx.shape))
    # data_idx = [data_idx]
    # print('data_idx = %s / %s' % (type(data_idx), data_idx))
    # data_idx.dtype,
    # data_idx_ = slice(data_idx.tolist())
    # print('data_idx_ = %s, %s' % (type(data_idx_), data_idx_))

    def get_mail_header(mailstr):
        mailstr_ = mailstr.split('\n')
        mailstr_header = '\n'.join(mailstr_[:5])
        print('mailstr_header = %s' % mailstr_header)
        return mailstr_header
    
    # data = twenty_train.data[:2]
    data = [twenty_train.data[data_idx_] for data_idx_ in data_idx]
    data = [get_mail_header(twenty_train.data[data_idx_]) for data_idx_ in data_idx]

    print('data = %s' % (data))
    # sys.exit(0)
    
    # # chop data
    # data_ = []
    # for i, data_i in enumerate(data):
    #     print('data_%d with len = %s' % (i, len(data_i)))
    #     c_base = 0
    #     c_len = len(data_i)
    #     while c_base < len(data_i):
    #         ranlen = min(20, np.random.randint(c_base, c_len))
    #         data_.append(data_i[:ranlen])
    #         c_base += ranlen

    
    data_ = data
            
    # train cv
    cv = CountVectorizer()
    data_cv = cv.fit_transform(data_)
    print(data_cv.shape)
    print(data_cv[0])
    print(dir(cv))

    # idf is downscale globally common words
    tf = TfidfTransformer(use_idf=False).fit(data_cv)
    data_tf = tf.transform(data_cv)
    
    # build the graph based in cv codes

    ## plot frequencies

    ## plot superimposed transform images max size embedding

    ## plot scatter
    ## plot scatter with edges
    ## plot scatter reduced
    ## plot scatter reduced with edges
    
    # swap codes
    # do fancy stuff

    def makeqtapp():
        app = QtGui.QApplication([])
        # Enable antialiasing for prettier plots
        # pg.setConfigOptions(antialias=True)
        
        # qm = QtGui.QMenu()
        # QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Return"), qm, app.quit())
        
        # exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        # exitAction.setShortcut('Ctrl+Q')
        # exitAction.setStatusTip('Exit application')
        # # exitAction.triggered.connect(QtGui.qApp.quit)
        # exitAction.triggered.connect(app)
        return app
        
    def makeplotwin():
        win = pg.GraphicsWindow(title="Basic plotting examples")
        win.resize(1000,600)
        win.setWindowTitle('pyqtgraph example: Plotting')

        return win

    app = makeqtapp()
    setup_interrupt_handling()
    win = makeplotwin()
    

    freqs = data_cv.sum(axis=0)
    freqs_size = freqs.shape[1]
    print('freqs: %s, %s' % (freqs.shape, type(freqs)))
    freqs_ = np.array(freqs).ravel()
    freqs_sort_idx = np.argsort(freqs_).tolist()
    print('freqs_sort_idx: %s' % (freqs_sort_idx))
    freqs_.sort()
    
    counts = data_cv.sum(axis=1)
    counts_max = np.max(counts)
    counts_ = np.array(counts).ravel()
    counts_sort_idx = np.argsort(counts_)
    counts_.sort()
    print('counts: %s, %s' % (counts.shape, type(counts)))

    p1 = win.addPlot(title="seq2graph: log-log token freqs")
    p1.plot(y=freqs_[::-1])
    p1.setLogMode(True, True)
    # p1.plot(np.random.normal(size=100), pen=(255,0,0), name="Red curve")

    p2 = win.addPlot(title="seq2graph: token counts / seq")
    p2.plot(y=counts_[::-1])
    p2.setLogMode(True, True)
    
    print('type(twenty_train.data[0])) = %s' % type(twenty_train.data[0]))
    # print('cv.tokenizer = %s' % (cv.tokenizer))

    sl = slice(0, 100)
    counts_max_2 = np.max([len(d_) for d_ in data])
    base_img = np.ones((counts_max_2, freqs_size))
    bla = base_img.copy()

    # app2 = makeqtapp()
    win2 = makeplotwin()
    win2.show()
    p2_1 = win2.addPlot(title='smp/playground: language tree')
    
    print('accumulating code sequence matrices')
    for s, seq in enumerate(data):
        seq_ = seq.split('\n')
        seq = ' '.join(seq_)
        seq_ = seq.split('\t')
        seq = ' '.join(seq_)
        seq_ = seq.split(':')
        seq = ''.join(seq_)
        seq_ = seq.split(' ')
        print('seq_%d = %s\nseq_%d_=%s' % (s, seq, s, seq_))
        d_0_code_seq = cv.transform(seq_)
        d_ = d_0_code_seq.todense()
        print('s = %s, d_s = %s, counts_max = %s, counts_max_2 = %s, freqs_size = %s, d_0_code_seq = %s' % (s, d_.shape, counts_max, counts_max_2, freqs_size, d_0_code_seq.shape))
        bla[0:d_0_code_seq.shape[0]] += d_

        x_ = [] # [0] * freqs_size
        y_ = []
        cnts_ = []
        for s_, tok in enumerate(d_0_code_seq):
            y = np.argmax(tok)
            # y__ = freqs_sort_idx[::-1].index(y)
            y__ = y
            print('tok = %s, y = %s, y__ = %s' % (tok, y, y__))
            x_.append(s_)
            y_.append(y__)

        x__ = np.array(x_)
        y__ = np.array(y_)

        # x___ = x__ + np.random.uniform(-1, 1, x__.shape) * 0.3
        # y___ = y__ + np.random.uniform(-1, 1, y__.shape) * 0.3
        x___ = x__ + np.random.normal(0, 1, x__.shape) * 0.3
        y___ = y__ + np.random.normal(0, 1, y__.shape) * 0.3
            
        colrgb = np.random.uniform(0, 255, (4,))
        c_1 = float((counts_max_2 - len(x_))/counts_max_2)
        c_2 = 255
        # colrgb *= 3
        # c_ = int(c_ * 255)
        # colrgb = [c_, c_, c_, 50]
        colrgb[:3] = c_2
        colrgb[3] = 50 # int(c_1)
        pen_ = pg.mkPen(colrgb.astype(np.int))
        # p2_1.plot(y_, x_, pen=pen_, symbol='o', symbolPen=pen_, symbolBrush=pg.mkBrush(colrgb))
        p2_1.plot(y___, x___, pen=pen_, symbol='o', symbolPen=pen_, symbolBrush=pg.mkBrush(colrgb))
        # pg.QtGui.QApplication.processEvents()
        app.processEvents()
        
        # print('bla = %s' % (bla.shape,))

    # plt.pcolormesh(np.log(bla), cmap=plt.get_cmap('Greys'))
    # plt.pcolormesh(bla, cmap=plt.get_cmap('Greys'))
    # plt.imshow(np.log(bla), cmap=plt.get_cmap('Greys'))
    # plt.colorbar()
    # plt.show()

    # ui = VideoTemplate.Ui_MainWindow()
    # ui.setupUi(win2)
    # riw = RawImageWidget()
    # ui.stack.addWidget(riw)
    # print('plotting image')
    # # pg.image(np.log(bla))
    # useLut = None
    # useScale = None
    # riw.setImage(bla, lut=useLut, levels=useScale)    

    # p2 = win.addPlot(title="Multiple curves")
    # p2.plot(np.random.normal(size=100), pen=(255,0,0), name="Red curve")
    # p2.plot(np.random.normal(size=110)+5, pen=(0,255,0), name="Green curve")
    # p2.plot(np.random.normal(size=120)+10, pen=(0,0,255), name="Blue curve")

    # # plot stuff
    # x = np.random.normal(size=1000)
    # y = np.random.normal(size=1000)
    # pg.plot(x, y, pen=None, symbol='o')  ## setting pen=None disables line drawing

    # x = np.arange(1000)
    # y = np.random.normal(size=(3, 1000))
    # plotWidget = pg.plot(title="Three plot curves")
    # for i in range(3):
    #     plotWidget.plot(x, y[i], pen=(i,3))  ## setting pen=(i,3) automaticaly creates three different-colored pens

    # pg.show()

    win.show()
    # QtGui.QApplication.instance().exec_()    
    app.exec_()    
        
def main(args):
    if args.mode == 'tree':
        main_tree(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='tree', help='Main execution mode [tree]')

    args = parser.parse_args()
    
    main(args)
