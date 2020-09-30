import networkx as nx
import numpy as np

from scipy.special import logsumexp


class SyntheticTree:
    def __init__(self, k, d, algorithm, tau):
        self._k = k
        self._d = d
        self._algorithm = algorithm
        self._tau = tau

        self._tree = nx.balanced_tree(k, d, create_using=nx.DiGraph)
        for e in self._tree.edges:
            self._tree[e[0]][e[1]]['weight'] = np.random.rand()
            self._tree[e[0]][e[1]]['N'] = 0
            self._tree[e[0]][e[1]]['Q'] = 0.

        for n in self._tree.nodes:
            self._tree.nodes[n]['N'] = 0
            self._tree.nodes[n]['V'] = 0.

        self.leaves = [x for x in self._tree.nodes() if
                       self._tree.out_degree(x) == 0 and self._tree.in_degree(x) == 1]
        for leaf in self.leaves:
            mean = 0.
            for path in nx.all_simple_edge_paths(self._tree, 0, leaf):
                for edge in path:
                    mean += self._tree[edge[0]][edge[1]]['weight']
            self._tree.nodes[leaf]['mean'] = mean

        self.optimal_v_root = self._solver()

        self.state = None

        self.reset()

    def reset(self, state=None):
        if state is not None:
            self.state = state
        else:
            self.state = 0

        return self.state

    def step(self, action):
        edges = [e for e in self._tree.edges(self.state)]
        self.state = edges[action][1]

        return self.state

    def rollout(self, state):
        return np.random.normal(self._tree.nodes[state]['mean'])

    @property
    def tree(self):
        return self._tree

    def _solver(self, node=0):
        if self._algorithm == 'uct':
            max_mean = 0.
            for leaf in self.leaves:
                max_mean = max(self._tree.nodes[leaf]['mean'], max_mean)

            return max_mean
        elif self._algorithm == 'ments':
            successors = [n for n in self._tree.successors(node)]
            if successors[0] in self.leaves:
                return self._tau * logsumexp(
                    [self._tree.nodes[x]['mean'] / self._tau for x in self._tree.successors(node)]
                )
            else:
                return self._tau * logsumexp(
                    [self._solver(n) / self._tau for n in self._tree.successors(node)]
                )
        else:
            raise ValueError
