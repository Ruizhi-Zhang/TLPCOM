import networkx as nx
import numpy as np
import pickle
import re

def load_data_as_graph(path, weight_idx=2, time_idx=3):

    edges = []
    with open(path) as f:
        next(f)
        for line in f:
            tokens = re.split(r'\s+', line)
            u = int(tokens[0])
            v = int(tokens[1])
            time = int(tokens[time_idx])
            if weight_idx:
                weight = int(tokens[weight_idx])

                # add edge
                edges.append((u, v, {'weight': weight, 'time':time}))
            else:
                edges.append((u, v, {'time': time}))

    g = nx.MultiDiGraph()
    g.add_edges_from(edges)

    print(len(g.edges))

    return g


def get_negative_edge(g, first_node=None):
    if first_node is None:
        first_node = np.random.choice(g.nodes())
    possible_nodes = set(g.nodes())
    neighbours = [n for n in g.neighbors(first_node)] + [first_node]
    possible_nodes.difference_update(neighbours)
    second_node = np.random.choice(list(possible_nodes))
    edge = (first_node, second_node, {'weight':1, 'time': None})
    return edge


def create_embedding_and_training_data_old(g, train_edges_fraction=0.75):
    edges = sorted(g.edges(data=True), key=lambda x: x[2]['time'])
    num_edges = len(edges)

    # training edges
    num_train_edges = int(train_edges_fraction * num_edges)
    train_edges = edges[:num_train_edges]

    # link prediction positive edges
    pos_edges = edges[num_train_edges:]
    neg_edges = []
    for i in range(len(pos_edges)):
        n_edge = get_negative_edge(g)
        neg_edges.append(n_edge)

    return train_edges, pos_edges, neg_edges


def create_embedding_and_training_data(g, train_edges_fraction, trainning_ratio):
    nodes = g.nodes()
    train_edges = []
    train_neg = []
    pos_edges = []
    neg_edges = []

    for node in nodes:
        edges_of_node = []
        for e in g.edges(node, data=True):
            edges_of_node.append(e)

        edges_of_node = sorted(edges_of_node, key=lambda x: x[2]['time'])
        num_edges = len(edges_of_node)

        # training edges per node
        num_train_edges = int(train_edges_fraction * num_edges)
        train_edges.extend(edges_of_node[:num_train_edges])

        # link prediction positive edges
        pos_edges.extend(edges_of_node[num_train_edges:])

    for i in range(len(pos_edges)):
        n_edge = get_negative_edge(g)
        neg_edges.append(n_edge)

    train_edges = sorted(train_edges, key=lambda x: x[2]['time'])
    ratio = int(len(train_edges) * (1 - trainning_ratio))
    train_edges = train_edges[ratio:]

    for i in range(len(train_edges)):
        n_edge = get_negative_edge(g)
        train_neg.append(n_edge)

    return train_edges, train_neg, pos_edges, neg_edges


def main():

    path = 'D:/.../.../.../contact.edges'
    network_g =  load_data_as_graph(path=path, weight_idx= 2, time_idx=3)
    embedding_edges, train_neg, pos_edges, neg_edges = create_embedding_and_training_data(network_g, train_edges_fraction=0.75,trainning_ratio = 1)


    save_path = 'D:/.../.../.../contact/'
    with open(save_path + 'embedding_edges', 'wb') as f:
        pickle.dump(embedding_edges, f)
    with open(save_path + 'pos_edges', 'wb') as f:
        pickle.dump(pos_edges, f)
    with open(save_path + 'neg_edges', 'wb') as f:
        pickle.dump(neg_edges, f)

if __name__ == '__main__':
    main()
