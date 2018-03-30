"""recursive tree walker

.. moduleauthor:: Oswald Berthold, 2018

**Yartwalk** - yet another recursive tree walker. The business of
recursively walking tree structures seems important in the dictionary
/ graph universe by number of occurrence.

**Use cases**:
- smp_graphs: walk nested graphs
- firefox session as json dict: analyze unknown dict, specify
  extraction request, extract with or without preserving nesting
  structure (ravel/flatten in numpy lingo; graphs as sparse matrices)
- structured data analysis
-- get xml, json, hdf5, dict, etc with unknown structure
-- try to extract this structural info from walking
-- take element at level i and apply configured handlers on elem_i
-- default handler: recur, if elem_i is composite, recur on it, else
   nothing
-- handlers: print elem, search for particular fields or strings,
   aggregate new structure
-- todo: need aggregation / return mechanics: init dict at level 0,
   pass in and out
-- todo: idea: match with graph tools, parse dict into graph
   straightaway, then use graph tools to analyze the structure
-- todo: idea: tasks: compress the graph: known substructures, nesting
   of identical substructures, explicit substructures of specific
   fieldnames, keys, etc, similarity of pure structure

**See also**: smp_graphs/graph,common,...; socnet; 
"""

from __future__ import print_function

from collections import OrderedDict
import argparse
import json

def get_filep(filename, mode='r'):
    f = None
    try:
        f = open(filename, mode)
    except Exception as err:
        print('get_filep: opening file %s failed with error = %s' % (args.filename, err))

    assert f is not None, 'Loading file somehow failed.'
    return f

def load_data_json(filename):
    f = get_filep(filename, 'r')
    d = json.load(f)
    d_bytes = f.tell()
    return d, d_bytes

def load_data_xml(filename):
    return None, None

def load_data(filename, filetype='json'):
    load_data_funcs = {
        'json': load_data_json,
        'xml': load_data_xml,
    }

    return load_data_funcs[filetype](filename)

def h_geturls(*args, **kwargs):
    # print('h_geturls args = %s' % (args))
    if len(args) < 1 and len(kwargs) < 1: return None

    # for arg in args:
    # with args[0] as arg:
    arg = args[0]
    if True:
        if type(arg) not in [str, unicode]: return None
        # if arg.count('url') > 0:
        if 'url' in arg:
            print('url: %s' % (arg, ))

def h_geturlsfromtabs(*args, **kwargs):
    # print('h_geturls args = %s' % (args))
    if len(args) < 1 and len(kwargs) < 1: return None

    # for arg in args:
    # with args[0] as arg:
    arg = args[0]
    if True:
        if type(arg) not in [dict, OrderedDict]: return None
        if 'window' in arg: print ('window = %s' % (arg,))
        # if arg.count('url') > 0:
        if 'url' in arg:
            # print('url %s' % (arg.keys(), ))
            # print('url')
            # print('  url = %s' % (arg['url']))
            print(arg['url'])
            # if 'originalURI' in arg:
            #     print('  originalURI = %s' % (arg['originalURI']))
            
def h_printer(*args, **kwargs):
    verbose = kwargs['verbose']
    if not verbose: return
    tree = args[0]
    level = kwargs['level']
    spacer = ' ' * 4 * level
        
    if iscomposite(tree):
        print(
            'recursive_walker%s[%d] v = %s, |v| = %d, type(v) = %s' % (
                spacer, level, tree, len(tree), type(tree)))
    else:
        print(
            'recursive_walker%s[%d] v = %s, type(v) = %s' % (
                spacer, level, tree, type(tree)))
    # return None

def h_tree(*args, **kwargs):
    pass
    
def list2dict(l):
    assert iscomposite(l), 'Object l is not composite with type = %s' % (type(l))
    if type(l) in [dict, OrderedDict]: return l
    return dict(zip(range(len(l)), l))
    
def iscomposite(x):
    # return type(x) in [dict, list, OrderedDict, str, unicode]
    return type(x) in [dict, list, OrderedDict]

def h_recurse(*args, **kwargs):
    tree = args[0]
    level = kwargs['level']
    handlers = kwargs['handlers']
    verbose = kwargs['verbose']
    
    # a tree leaf
    if not iscomposite(tree): return
        
    # a tree branch
    for i, k in enumerate(list2dict(tree)):
        # v = tree[k]
        recursive_walker(tree[k], level+1, handlers, verbose)
               
handlers = {
    'tree': h_tree,
    'recurse': h_recurse,
    'printer': h_printer,
    'geturls': h_geturls,
    'getsess': h_geturlsfromtabs,
}

def recursive_walker(tree, level=0, handlers=[], verbose=True):
    """recursive walker

    Recursively walk any tree-like structure such as dictionaries and
    lists. Common use cases are json or xml data.
    """
    # assert iscomposite(tree), 'recursive_walker: arg tree has wrong type = %s' % (type(tree))
    spacer = ' ' * 4 * level
    # if verbose: print('recursive_walker%s[%02d]' % (spacer, level))

    # a tree leaf
    # if not iscomposite(tree):
    
    for h in handlers: h(tree, level=level, handlers=handlers, verbose=verbose)
    # return None

def main(args):
    assert args.filename is not None, 'No filename given, exiting'

    # FIXME: file type?
    if args.filename.endswith('.json'):
        filetype = 'json'
    elif args.filename.endswith('.xml'):
        filetype = 'xml'
        
    d, d_bytes = load_data(args.filename, filetype)

    assert d is not None, 'No data loaded, d = %s, d_bytes = %s' % (d, d_bytes)

    print('Loaded data of type = %s, bytes = %s, size = %s' % (type(d), d_bytes, len(d)))

    print('Walking data')

    recursive_walker(d, 0, [handlers[k] for k in handlers if k in args.handlers], True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default=None, help='File to load and scan [None]')
    parser.add_argument('-a', '--handlers', type=str, default='', help='Handlers to apply to leaves [\'\'], one of %s' % (list(handlers)))
    parser.add_argument('-t', '--filetype', type=str, default='json', help='Type of data in the file [json], one of [json, xml, pickle]')

    args = parser.parse_args()

    main(args)
