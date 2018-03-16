"""network analysis

See also: socnetv
"""
import argparse
import networkx as nx
import matplotlib.pyplot as plt

from socnet_data import data as ppl

def main_simple(args):
    # print "main_simple, args = {0}".format(args)

    G = nx.MultiDiGraph()
    G.add_nodes_from(ppl)

    # print "G", G.nodes
    nx.draw_networkx(G)
    plt.show()
    
    
def main(args):
    if args.mode in modes:
        modes[args.mode](args)

modes = {
    'simple': main_simple
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest='mode', default='simple', type=str, help='Exec mode [simple]')

    args = parser.parse_args()

    main(args)
