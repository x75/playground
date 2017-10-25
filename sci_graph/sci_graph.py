"""use graph exploration for text production

Oswald Berthold 2017
"""
import argparse, re

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def get_data_raw():
    data_raw = {}
    data_raw['1'] = """science of intelligence,
    principles, exploration, exploitation, complexity, reduction, decision making,
    time, processes, 
    interaction, environmental, social,
    theory of mind,
    heuristics, nice strategies,
    anticipation, prediction,
    flow, intrinsic motivation,
    diversity, variability,
    closed-loop learning,
    introspection, reflectivity, cognition,
    pillars, metabolism, reproduction,
    evolution, population, diversity, heredity, selection,
    deductive, mathematical, biology,
    social life of electrons,
    decomposition,
    intelligence,
    open-ended learning,
    short-term, consolidation, memory,
    21st century,
    evolutionary robotics, developmental robotics,
    psychology, neuroscience, machine learning, optimization,
    control theory, reinforcement learning, cybernetics,
    embodiment, exploration, self-exploration, black-box optimization,
    representation learning, relational learning, error-based learning,
    the reward prediction hypothesis of dopamine, exploitation,
    stochastic gradient descent, self-organization, learning rules,
    correlational learning, measures, data,
    tappings,
    internal models, developmental models,
    environment, agents, predictive coding,"""
    return data_raw
    
def main_view(args):
    """
    """
    
    data_raw = get_data_raw()
    data_raw_str = re.sub('\n', '', data_raw['1'])
    data_raw_str = re.sub(', +', ', ', data_raw_str)
    
    data_raw_ = [d.lstrip() for d in data_raw_str.split(",")[:-1]]
    print "data_raw_", data_raw_

    # init graph
    G = nx.MultiDiGraph()

    # add nodes
    for i, item in enumerate(data_raw_):
        kwargs = {'label': item}
        G.add_node(i, **kwargs)

    # add edges
    # FIXME: how to get good edges algorithmically?
        
    # draw the graph
    nx.draw_networkx(
        G, node_color = 'k', alpha = 0.33,
        labels = dict([(n, G.node[n]['label']) for n in G.nodes()]))

    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type = str, default = 'view', help = 'script exec mode [view]')

    args = parser.parse_args()

    if args.mode in ['view']:
        main_view(args)
    else:
        print "scioi_graph.py: exiting on unknown mode = %s" % (args.mode, )
