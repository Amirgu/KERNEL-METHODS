import networkx
import pickle
import pandas as pd
import numpy as np
from algorithms import compute_support, g_span,load_graphs,DFS2G,subgraph_isomorphisms

import networkx as nx

#### USED TO CONVERTS A LIST OF NETWORKX GRAPH TO A TEXT FILE WHICH COULD BE USED BY ALGORITHMS.py(GSPAN)
def write_nodes_and_edges_with_labels_to_file(graphs, filename):
    with open(filename, 'w') as file:
        for i, graph in enumerate(graphs):
            # Write a separator line
            file.write(f"t # {i }\n")
            # Write the nodes
            for node in graph.nodes():
                label = graph.nodes[node]['labels']
                file.write(f"v {node} {label[0]}\n")
            # Write the edges
            for u, v in graph.edges():
                label = graph[u][v]['labels']
                file.write(f"e {u} {v} {label[0]}\n")


### with open('training_data.pkl','rb') as f:
###       G=pickle.load(f)
### write_nodes_and_edges_with_labels_to_file(G,'/Gn.txt')

####LOADING THE CORRESPONDING TEXT FILE 
graphs = load_graphs('projet/data/Gn.txt')
gp = load_graphs('projet/data/Gpositive.txt') ### Graphs in class 1
gn = load_graphs('projet/data/Gnegative.txt') ### Graphs in class 0

### MINING FREQUENT SUBGRAPHS
##extensions = []
##g_span([],graphs, min_sup= 1200, extensions=extensions)

def score(C, gp, gn):
    """
    Computes the score of a subgraph C.

    Args:
    - C: subgraph
    - gp: graph P
    - gn: graph N

    Returns:
    - score: subgraph score
    """
    a = compute_support(C, gp)[0] / len(gp)
    b = compute_support(C, gn)[0] / len(gn)
    score = np.abs(a - b)
    return score

def sort(extensions):
    """
    Sorts a list of subgraph extensions by their scores.

    Args:
    - extensions: list of subgraph extensions

    Returns:
    - sorted_extensions: sorted list of subgraph extensions by score in this format ( score(sg),index in extensions)
    """
    sorted_extensions = sorted([(score(c), i) for i, c in enumerate(extensions)], key=lambda x: x[0])
    return sorted_extensions

def find_first_tuple_index(sorted_list, number):
    left = 0
    right = len(sorted_list) - 1
    while left <= right:
        mid = (left + right) // 2
        mid_val = sorted_list[mid][0]
        if mid_val < number:
            if sorted_list[mid + 1][0] >= number:
                return mid
            left = mid + 1
        else:
            right = mid - 1
    return -1

def MDFSandLDFS(a,L):
    i0=find_first_tuple_index(L,a)
    LDFS=L[:i0+1]
    MDFS=L[i0+1:]
    return LDFS,MDFS


with open('extensionsforS800.pkl', 'rb') as f:
    # use pickle to dump the variable to the file
    extensions=pickle.load( f)

with open('LDFS=800.pkl', 'rb') as f:
    # use pickle to dump the variable to the file
  LDFS=pickle.load(f)
with open('MDFS=800.pkl', 'rb') as f:
    # use pickle to dump the variable to the file
  MDFS=pickle.load(f)



def Filtersg(G, sg):
    """
    Finds the subgraphs in `sg` that are isomorphic to `G`.

    Args:
    - G: graph
    - sg: set of subgraphs

    Returns:
    - s: list of subgraphs in `sg` that are isomorphic to `G`
    """
    s = []
    for i in range(len(sg)):
        if len(subgraph_isomorphisms(sg[i], G)) > 0:
            s.append(sg[i])
    return s







def remne(G1, nodes, edges):
    """
    Removes nodes and edges from the graph `G1`.

    Args:
    - G1: graph
    - nodes: list of nodes to be removed
    - edges: list of edges to be removed

    Returns:
    - T: graph after removing nodes and edges
    """
    T = G1
    DFS = T.print_graph()
    A = [edge for edge in DFS if edge[3] not in nodes and edge[2] not in nodes]
    A = [edge for edge in A if edge not in edges]
    return DFS2G(A)

def converts(G1):
    """
    Converts a graph `G1` to a NetworkX graph object.

    Args:
    - G1: graph

    Returns:
    - G: NetworkX graph object
    """
    DFS = G1.print_graph()
    G = nx.Graph()
    for edge in DFS:
        G.add_node(edge[0], labels=edge[2])
        G.add_node(edge[1], labels=edge[3])
        G.add_edge(edge[0], edge[1], labels=edge[4])
    return G


def compute_isomorphisms(C, G):
    """
    Computes the isomorphisms of a subgraph `C` in a graph `G`.

    Args:
    - C: subgraph
    - G: graph

    Returns:
    - s: list of isomorphisms found
    """
    iso = subgraph_isomorphisms(C, G)
    n = len(iso)
    G_C = G.print_graph()
    s = []
    for i in range(n):
        T = C.copy()
        d = []
        mapping = dict(iso[i])
        for j in range(len(iso[i]) - 1):
            T[j] = list(T[j])
            u, v, *labels = T[j]
            T[j] = (mapping[u], mapping[v], *labels)
            # EDGE
            D = (T[j][1], T[j][0], T[j][3], T[j][2], T[j][4])
            if T[j] in G_C:
                s.append(T[j])
            elif D in G_C:
                s.append(D)
    return s


def enleve(C, G):
    """
    Removes subgraphs a list `C` of subgraphs from graph `G`.

    Args:
    - C: list of subgraphs
    - G: graph

    Returns:
    -  Graph with the subgraphs removed
    """
    A = G.print_graph()
    r = A.copy()
    B = C.copy()
    for j in range(len(C)):
        T = compute_isomorphisms(B[j], G)
        r = set(r) - set(T)

    # Returns the graph with the subgraphs removed
    return set(A) - set(r)


def removeldfsmdfs(G, L, M):
    """
    give nodes and edges in L to remove from G if they were not overlapping with subgraphs M .

    Args:
    - G: graph
    - L: one subgraph
    - M: list of lists of subgraphs

    Returns:
    - nodes_leave: set of nodes to be removed from `G`
    - edges_leaves: set of edges to be removed from `G`
    """
    nodes_leave = set()
    edges_leaves = set()
    r = list(enleve(M, G))  # Removes the structure of M from G and converts to a list

    t = set(G.print_graph()) - set(r)
    t = list(t)

    e = DFS2G(t)

    for i in range(len(L)):
        Y = enleve([[L[i]]], e)
        edges_leaves.update(Y)

    nodesM = getnodes(r)
    nodesL = getnodes(L)

    nodes_leave = nodesL - nodesM.intersection(nodesL)

    return nodes_leave, edges_leaves


def getnodes(G):
    nodes=set()
    for i in range(len(G)):
        nodes.add(G[i][2])
        nodes.add(G[i][3])
    return nodes

### MAIN ALGORITHM 
def reconstruct(Dm, LDFS, MDFS, extensions):
    """
    Reconstructs graphs from a given list of graphs `Dm`, using subgraph structures
    `LDFS`, `MDFS`, and `extensions`.

    Args:
    - Dm: list of graphs to reconstruct 
    - LDFS: list of least discriminative subgraphs in the format from MDFSandLDFS function 
    - MDFS: list of most discriminative subgraphs in the format from MDFSandLDFS function 
    - extensions: list of subgraphs

    Returns:
    - Result: list of reconstructed Networkx graphs 
    """
    D = Dm.copy()
    Result = []
    r = [extensions[MDFS[k][1]] for k in range(len(MDFS))]
    for i in range(len(D)):
        for j in range(len(LDFS)):
            L = extensions[LDFS[j][1]]
            s = subgraph_isomorphisms(L, D[i])
            if len(s) > 0:
                m = Filtersg(D[i], r)
                nodes, edges = removeldfsmdfs(D[i], L, m)
                D[i] = remne(D[i], nodes, edges)
        Result.append(converts(D[i]))
        if i % 500 == 0:
            print('on est la ', i, '\n')
    return Result

###AFTER MINING SUBGRAPHS WITH gSPAN:
### L=sort(extensions)
### LDFS,MDFS= MDFSandLDFS(0.3,L)
### Dm=reconstruct(graphs,LDFS,MDFS,extensions)

###PART FOR INFREQUENT SUBGRAPHS (NOT USED)

def enleve2(C,G): ## G have to from class graph in algo
    ### C is list of lists 
   
    A=G.print_graph()
    r=A.copy()
    B=C.copy()
    for j in range(len(C)):
        
        
        
        T=compute_isomorphisms(B[j],G)
        
        
        
        
        r=set(r) - set(T)
    
    ##returns the graphs inter the wanted subgraphs ie the structure of which we can leave

    return  set(r)

def IS(extensions,graphs):
    s=set()
    for i in range(len(graphs)):
        ext=Filtersg(graphs[i],extensions)
        T=graphs[i].print_graph()
        r=enleve2(ext,graphs[i])
        s.update(r)
    d=list(s)
    unique_tuples = {}
    for tup in s:
    # Extract the last three elements as a tuple
        key = tuple(tup[-3:])
    # Add the tuple as a key to the dictionary with the original tuple as the value
        unique_tuples[key] = tup

# Convert the dictionary back to a list of tuples
    output_list = list(unique_tuples.values())

    return output_list 


def scoreIS(IS):
    g=[]
    for i in range(len(IS)):
        g.append((score([IS[i]]),i))
   
    return sorted(g, key=lambda x: x[0])

