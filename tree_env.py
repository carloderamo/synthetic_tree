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
        random_weights = np.random.rand(len(self._tree.edges))
        for i, e in enumerate(self._tree.edges):
            self._tree[e[0]][e[1]]['weight'] = random_weights[i]
            self._tree[e[0]][e[1]]['N'] = 0
            self._tree[e[0]][e[1]]['Q'] = 0.

        for n in self._tree.nodes:
            self._tree.nodes[n]['N'] = 0
            self._tree.nodes[n]['V'] = 0.

        self.leaves = [x for x in self._tree.nodes() if
                       self._tree.out_degree(x) == 0 and self._tree.in_degree(x) == 1]

        self._compute_mean()
        means = np.array([self._tree.nodes[n]['mean'] for n in self.leaves])
        means = (means - means.min()) / (means.max() - means.min()) if len(means) > 1 else [0.]
        for i, n in enumerate(self.leaves):
            self._tree.nodes[n]['mean'] = means[i]

        self.max_mean = 0.
        for leaf in self.leaves:
            self.max_mean = max(self._tree.nodes[leaf]['mean'], self.max_mean)

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
        return np.random.normal(self._tree.nodes[state]['mean'], scale=.05)

    @property
    def tree(self):
        return self._tree

    def _compute_mean(self, node=0, weight=0):
        if node not in self.leaves:
            for e in self._tree.edges(node):
                self._compute_mean(e[1],
                                   weight + self._tree[e[0]][e[1]]['weight'])
        else:
            self._tree.nodes[node]['mean'] = weight

    def _solver(self, node=0):
        if self._algorithm == 'uct' or self._algorithm == 'rents':
            return self.max_mean
        else:
            successors = [n for n in self._tree.successors(node)]
            if self._algorithm == 'ments':
                if successors[0] in self.leaves:
                    return self._tau * logsumexp(
                        [self._tree.nodes[x]['mean'] / self._tau for x in self._tree.successors(node)]
                    )
                else:
                    return self._tau * logsumexp(
                        [self._solver(n) / self._tau for n in self._tree.successors(node)]
                    )
            elif self._algorithm == 'tents':
                def sparse_max(means_tau):
                    temp_means_tau = means_tau.copy()
                    sorted_means = np.flip(np.sort(temp_means_tau))
                    kappa = list()
                    for i in range(1, len(sorted_means) + 1):
                        if 1 + i * sorted_means[i-1] > sorted_means[:i].sum():
                            idx = np.argwhere(temp_means_tau == sorted_means[i-1]).ravel()[0]
                            temp_means_tau[idx] = np.nan
                            kappa.append(idx)
                    kappa = np.array(kappa)

                    sparse_max = means_tau[kappa] ** 2 / 2 - (
                            means_tau[kappa].sum() - 1) ** 2 / (2 * len(kappa) ** 2)
                    sparse_max = sparse_max.sum() + .5

                    return sparse_max

                if successors[0] in self.leaves:
                    means = np.array([self._tree.nodes[x]['mean'] for x in self._tree.successors(node)])
                    return self._tau * sparse_max(means / self._tau)
                else:
                    return self._tau * sparse_max(np.array(
                        [self._solver(x) / self._tau for x in self._tree.successors(node)])
                    )
            else:
                raise ValueError
