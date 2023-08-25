import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def generate_dag(nvertices, density):
    nedges = round(density*nvertices*(nvertices-1)/2)

    all_edges = [(i, j) for i in range(nvertices-1) for j in range(i+1, nvertices)]
    np.random.shuffle(all_edges)
    edges = all_edges[:nedges]
    
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G

def draw_dag(G, root_node=0):
    pos = {}
    x_pos = single_source_longest_dag_path_length(G, 0)
    heads = [v1 for (v1, _) in G.edges]
    longest_length = max(list(x_pos.values()))
    for node in G.nodes:
        if node not in heads:
            x_pos[node] = longest_length
    y_pos = {i: [] for i in range(len(G.nodes))}
    for k, v in x_pos.items():
        y_pos[v].append(k)


    for i, (node, x) in enumerate(x_pos.items()):

        pos[node] = (x, len(y_pos[x])/2-y_pos[x].index(node))

    nx.draw(G, pos, with_labels=True, node_size=200, node_color='lightblue', font_weight='bold')
    plt.show()

def single_source_longest_dag_path_length(graph, s):
    dist = dict.fromkeys(graph.nodes, -float('inf'))
    dist[s] = 0
    topo_order = nx.topological_sort(graph)
    for n in topo_order:
        for s in graph.successors(n):
            if dist[s] < dist[n] + 1:
                dist[s] = dist[n] + 1
    return dist

    # for node in G.nodes
def single_source_shortest_dag_path_length(G, root_node=0):
    shortest_path = {i: [] for i in range(len(G.nodes))}
    nodes_remove = []
    for node in G.nodes:
        try:
            shortest_path[node] = nx.shortest_path_length(G, source=root_node, target=node)
        except nx.exception.NetworkXNoPath:
            nodes_remove.append(node)

    print(len(nodes_remove))
    G.remove_nodes_from(nodes_remove)
 
    return shortest_path

def remove_opposite_dir(G):
    shortest_path = single_source_shortest_dag_path_length(G)
    edges_reverse = []
    cycle_remove = []
    for v1, v2 in G.edges:
        
        if shortest_path[v1]>shortest_path[v2]:
            edges_reverse.append((v1,v2))
        elif shortest_path[v1]==shortest_path[v2]:
            if (v1, v2) in G.edges and (v1, v2) in G.edges:
                cycle_remove.append(tuple(sorted([v1,v2])))

    
    G.remove_edges_from(edges_reverse)
    G.remove_edges_from(cycle_remove)
    G.add_edges_from(list(map(lambda edge: (edge[1],edge[0]), edges_reverse)))
    single_source_shortest_dag_path_length(G)
    
    return G

if __name__ == '__main__':
    nnodes = 0
    while nnodes<60:
        G:nx.Graph = nx.gnp_random_graph(100, 0.01, directed=True)
        G = remove_opposite_dir(G)
        nnodes = len(G.nodes)
    draw_dag(G)