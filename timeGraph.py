import networkx as nx
import numpy as np
import pickle
import random
import time
from scipy.stats import wasserstein_distance

class TimeGraph:
    def __init__(self, edgelist, time_prop_name):
        self.G = nx.MultiDiGraph()
        self.G.add_edges_from(edgelist)
        self.time_prop_name = time_prop_name
        self.edges = list(self.G.edges(data=True))

        self.beta = None # number of temporal context windows
        self.D = None # embedding dimension
        self.L = None # max walk length
        self.omega = None # min walk length / context window size for skip gram

        self.all_time_walks = None
        self.num_nodes = len(list(self.G.nodes()))
        self.num_edges = len(self.edges)
        self.dict_distribution = self.nodes_prob_distribution()

    def nodes_prob_distribution(self):
        dict_distribution = {}
        for node in self.G.nodes():
            curr_node_neighbors = list(self.G.neighbors(node))
            distribution = [self.G.out_degree(neighbor) for neighbor in curr_node_neighbors]
            dict_distribution[node] = distribution
        return dict_distribution

    def set_temporal_walk_params(self, beta, D, omega, L=80):
        self.beta = beta
        self.D = D
        self.L = L
        self.omega = omega

    def sample_edge(self):
        num_edges = len(self.edges)
        choice = np.random.choice(num_edges)
        return self.edges[choice]

    def generate_ctdne_walks(self):
        all_time_walks = []

        C = 0
        counter = 0
        while self.beta - C > 0:
            u, v, prop = self.sample_edge()
            t = prop[self.time_prop_name]

            walk_t = self._temporal_walk(start_edge=(u, v), t=t, C=self.omega + self.beta - C - 1)

            if (walk_t is not None) and (len(walk_t) >= self.omega):
                all_time_walks.append(walk_t)
                C = C + (len(walk_t) - self.omega + 1)

            counter += 1
            if (counter + 1) % 1000 == 0:
                print('Loop ran for ', counter + 1, 'times!\t Current C = ', C)

        self.all_time_walks = all_time_walks

    # wasserstein_distance-based sampling
    def _temporal_walk(self, start_edge=None, t=None, C=0):
        G = self.G
        prop_name = self.time_prop_name

        if start_edge:
            path = [start_edge[0], start_edge[1]]
        else:
            raise ValueError('start_edge should not be None.')

        curr_node = start_edge[1]

        for p in range(1, min(self.L, C)):
            neighbor_candidates = []
            for u, v, prop in G.out_edges(curr_node, data=True):
                if prop[prop_name] >= t:
                    neighbor_candidates.append((v, prop[prop_name]))

            # check if there are valid neighbors
            if len(neighbor_candidates) > 0:

                if not self.dict_distribution[curr_node]:
                    break
                else:
                    distance = [float('inf') if not self.dict_distribution[neighbor[0]]
                                else wasserstein_distance(self.dict_distribution[curr_node], self.dict_distribution[neighbor[0]])
                                for neighbor in neighbor_candidates
                                ]
                idx_next_node = np.argmin(distance)
                curr_node, t = neighbor_candidates[idx_next_node]
                # add new current node to path
                path.append(curr_node)
            else:
                break

        # MH-based sampling for node v
        spatial_nodes = []

        current_node = start_edge[1]
        while len(spatial_nodes) < 8:
            neighbors = list(G.neighbors(current_node))
            if len(neighbors) > 0:

                random_neighbor = random.sample(neighbors, 1)[0]
                p = np.random.uniform(0, 1)
                degree_curr_node = G.degree(current_node)
                degree_curr_random_nei = G.degree(random_neighbor)
                if p < degree_curr_node / degree_curr_random_nei:
                    spatial_nodes.append(random_neighbor)
                    current_node = random_neighbor
                else:
                    current_node = start_edge[1]
            else:
                break

        # MH-based sampling for node u
        spatial_nodes_u = []
        current_node = start_edge[0]
        while len(spatial_nodes_u) < 8:
            neighbors = list(G.neighbors(current_node))
            if len(neighbors) > 0:
                random_neighbor = random.sample(neighbors, 1)[0]
                p = np.random.uniform(0, 1)
                degree_curr_node = G.degree(current_node)
                degree_curr_random_nei = G.degree(random_neighbor)
                if p < degree_curr_node / degree_curr_random_nei:
                    spatial_nodes_u.append(random_neighbor)
                    current_node = random_neighbor
                else:
                    current_node = start_edge[0]
            else:
                break
        path = path + spatial_nodes + spatial_nodes_u


        return path


def main():

    path = 'D:/.../.../.../contact.time.walks'
    save_path = 'D:/.../.../.../contact/'


    with open(save_path + 'embedding_edges', 'rb') as f:
        embedding_edges = pickle.load(f)  # list

    timeG =TimeGraph(embedding_edges, 'time')

    R = 10
    N = timeG.num_nodes
    omega = 10
    L = 80
    beta = R * N * (L - omega + 1)

    print("Beta value:", beta)

    print("Started Walk...")

    timeG.set_temporal_walk_params(beta=beta, D=128, omega=omega, L=L)
    timeG.generate_ctdne_walks()

    ctdne_walks = timeG.all_time_walks

    with open(path, 'wb') as f:
        pickle.dump(ctdne_walks, f)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('总时间：', end - start)
