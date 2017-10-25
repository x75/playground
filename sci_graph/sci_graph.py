"""use graph exploration for text production

Oswald Berthold 2017
"""
import argparse, re

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def get_data_raw():
    data_raw = {}
    data = {}

    # data_raw['1'] = """
    # science of intelligence,
    # principles, exploration, exploitation, complexity, reduction, decision making,
    # time, processes, 
    # interaction, environmental, social,
    # ,
    # heuristics, nice strategies,
    # anticipation, prediction,
    # flow, intrinsic motivation,
    # diversity, variability,
    # closed-loop learning,
    # introspection, reflectivity, cognition,
    # pillars, metabolism, reproduction,
    # evolution, population, diversity, heredity, selection,
    # deductive, mathematical, biology,
    # social life of electrons,
    # decomposition,
    # intelligence,
    # open-ended learning,
    # short-term, consolidation, memory,
    # 21st century,
    # evolutionary robotics, developmental robotics,
    # psychology, neuroscience, machine learning, optimization,
    # control theory, reinforcement learning, cybernetics,
    # embodiment, exploration, self-exploration, black-box optimization,
    # representation learning, relational learning, error-based learning,
    # the reward prediction hypothesis of dopamine, exploitation,
    # stochastic gradient descent, self-organization, learning rules,
    # correlational learning, measures, data,
    # tappings,
    # internal models, developmental models,
    # environment, agents, predictive coding,"""
    
    data_raw['principles'] = """science, intelligence, principles, exploration, exploitation, complexity, anticipation, prediction, time, processes, interaction, mind, autonomy, autonomous, agent, population, society, social, diversity, variability, energy, metabolism, artificial, life, introspection, learning, adaptation, open-ended, lifelong, data, models, coding, curiosity, motivation, decision making, planning, optimality, adequacy, ecological, balance"""
    
    data_raw['fields'] = """neuroscience, statistics, artificial intelligence, cognitive robotics, psychology, economics, psychiatry, sociology, biology, chemistry, artificial life, control theory, cybernetics, medicine"""

    data_raw['methods'] = """theory of mind, complexity reduction, internal models, mixtures of experts, graphical models, closed-loop learning, bootstrapping, babbling, epsilon-greedy, reward prediction learning, reward prediction hypothesis of dopamine, intrinsic motivation, self-organization, confidence sampling, machine learning"""

    for k,v in data_raw.items():
        v = re.sub('\n', '', v)
        v = re.sub(', +', ', ', v)
        data_raw[k] = v.lstrip()
        data[k] = [d for d in data_raw[k].split(",")[:-1]]
    return data_raw, data

def get_subgraph_by_attr(G, attr, val):
    G_ = nx.MultiDiGraph()
    G_.add_nodes_from([(k, v) for k, v in G.nodes.data() if v.has_key(attr) and v[attr] == val])
    print "getting subgraph for", attr, val, "found", G_.nodes
    return G_

def get_nodes_by_attr(G, attr = None, val = None):
    if attr is None:
        return G.nodes(data = True), G
    else:
        G_ = get_subgraph_by_attr(G, attr, val)
        return G_.nodes(data = True), G_

def main_view(args):
    """
    """
    
    data_raw, data = get_data_raw()
    
    
    # print "data_raw_", data_raw_

    # init graph
    # G = nx.MultiDiGraph()
    G = nx.MultiGraph()
    G.name = "scigraph"

    # add nodes
    nodecnt = 0
    nodes_by_type = {}
    for datak, datav in data.items():
        nodes_by_type[datak] = [nodecnt, nodecnt]
        for i, item in enumerate(data[datak]):
            kwargs = {
                'label': item,
                'type': datak,}
            G.add_node(nodecnt, **kwargs)
            # print "added node %d = %s" % (i, G.node[i])
            nodecnt += 1
        nodes_by_type[datak][1] = nodecnt

    # add edges
    # FIXME: how to get good edges algorithmically?
        
    # draw the graph
    fig = plt.figure(figsize = (10, 10))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0,0])
    fig.suptitle(G.name)
    
    glayouts = [nx.random_layout, nx.circular_layout, nx.spring_layout, nx.shell_layout, ]
    gcolors = {'principles': 'k', 'fields': 'g', 'methods': 'r'}
    gscale = 1

    typecnt = 0
    xcenter = 0
    ycenter = 0
    node_type_start = 0
    # for datak, datav in data.items():
    for i_ in [0]:
        xcenter = np.cos((typecnt + 1) / float(len(data.keys())) * 2 * np.pi)
        ycenter = np.sin((typecnt + 1) / float(len(data.keys())) * 2 * np.pi)
        # print "data[%s] = %s" % (datak, data[datak])
        
        # G_nodes, G_ = get_nodes_by_attr(G, 'type', datak)
        # G_nodes, G_ = get_nodes_by_attr(G)
        G_nodes = G.nodes
        G_ = G

        # shells is list of list: layouts
        glayouts_shells = [range(nodes_by_type[nodetype][0], nodes_by_type[nodetype][1]) for nodetype in data.keys()]

        # shells is list of list: colors expanded and flattened
        gcolors_shells = [
            [
                gcolors[nodetype] for _ in range(nodes_by_type[nodetype][0], nodes_by_type[nodetype][1])
            ] for nodetype in data.keys()
        ]
        gcolors_shells = np.hstack(gcolors_shells)
        
        # print "glayouts_shells", glayouts_shells
        # print "gcolors_shells", gcolors_shells.shape
        
        # pos = glayouts[0](G_, center = [xcenter, ycenter])
        # pos = glayouts[2](G_, center = [xcenter, ycenter], scale = gscale)
        pos = glayouts[3](G_, glayouts_shells, center = [xcenter, ycenter], scale = gscale)

        gshell_xs = [0] * len(glayouts_shells)
        gshell_ys = [0] * len(glayouts_shells)
        for k,v in pos.items():
            gshell = 0
            while not k in glayouts_shells[gshell]:
                gshell += 1
                
            # pos[k] = (v[0] + (gshell * 3), v[1] + 0)
            dx = 0 # np.random.uniform(-1, 1) * 0.1 # 0
            dy = np.random.uniform(0.5, 1) * 0.01 * 50 # * len(glayouts_shells[gshell]) # 0
            gshell_xs[gshell] += dx
            gshell_ys[gshell] += dy
            if gshell_ys[gshell] > 5:
                gshell_ys[gshell] = 0
                gshell_xs[gshell] += 0.5
            
            pos[k] = (gshell * 2 + gshell_xs[gshell], gshell_ys[gshell])
            # pos[k] = (gshell * 3 + np.random.uniform(-1, 1) * 0.1, np.random.uniform(-1, 1) * 0.1 * len(glayouts_shells[gshell]))

        # print "typecnt = %d, datak = %s, G_nodes = %s" % (typecnt, datak, len(G_nodes))
        
        # create edges
        for edgecnt in range(100): # np.random.randint(20, 60)):
            gshell = np.random.randint(2)
            n1idx = len(glayouts_shells[gshell]) # G_.number_of_nodes()
            n2idx = len(glayouts_shells[gshell+1]) # G_.number_of_nodes()
            n1 = glayouts_shells[gshell][np.random.randint(n1idx)]
            n2 = glayouts_shells[gshell+1][np.random.randint(n2idx)]
            d = {'weight': np.exp(np.random.uniform(0, 1.0))}
            edge_ = (n1, n2, d)
            # edge_ = (n2, n1, d)
            G_.add_edges_from([edge_])
            # G_.add_edges_from()


        node_sizes = np.array([1 + np.power(d, 2) for n, d in nx.degree(G_)])
        print "node_sizes", node_sizes
            
        nx.draw_networkx_nodes(
            G_, pos = pos,
            ax = ax,
            node_size = 80 * gscale * node_sizes,
            node_color = gcolors_shells, alpha = 0.33,
        )

        # draw edges
        nx.draw_networkx_edges(
            G_, pos = pos,
            ax = ax,
            node_color = gcolors_shells, alpha = 0.2,
        )

        # Shift graph2
        xshift_labels = -0.05 * gscale
        yshift_labels = -0.05 * gscale

        pos_labels = {}
        # print "pos", pos
        for k,v in pos.items():
            pos_labels[k] = (v[0] + xshift_labels, v[1] + yshift_labels)
        
        # pos = layouts[2](G, center = [0.0, -0.1], scale = 10)
        nx.draw_networkx_labels(
            G_, pos = pos_labels,
            ax = ax,
            font_size = 8,
            labels = dict([(n, '%s-%d' % (G_.node[n]['label'], n, )) for n in G_.nodes()]),
        )

        typecnt += 1

    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    print "%s statistics"  % (G_.name, )
    print "\n    degree = %s" % (G_.degree(), )
    print "\n    degree_histogram = %s" % (nx.degree_histogram(G_), )
    print "\n    directed = %s" % (nx.is_directed(G_), )
    # print "\n    clustering = %s" % (nx.clustering(G_), )
        
    # ax2 = fig.add_subplot(gs[0,1])
    # ax2.bar(range(len(nx.degree_histogram(G_))), nx.degree_histogram(G_))

    
    plt.show()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type = str, default = 'view', help = 'script exec mode [view]')

    args = parser.parse_args()

    if args.mode in ['view']:
        main_view(args)
    else:
        print "scioi_graph.py: exiting on unknown mode = %s" % (args.mode, )
