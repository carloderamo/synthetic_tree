import numpy as np


class MCTS:
    def __init__(self, exploration_coeff, algorithm='uct'):
        self._exploration_coeff = exploration_coeff
        self._algorithm = algorithm

    def run(self, tree_env, n_simulations):
        v_hat = np.zeros(n_simulations)
        for i in range(n_simulations):
            tree_env.reset()
            v_hat[i] = self._simulation(tree_env)

        return v_hat

    def _simulation(self, tree_env):
        path = self._navigate(tree_env)
        leaf_node = tree_env.tree.nodes[path[-1][1]]
        leaf_node['V'] = (leaf_node['V'] * leaf_node['N'] +
                          tree_env.rollout(path[-1][1])) / (leaf_node['N'] + 1)
        tree_env.tree[path[-1][0]][path[-1][1]]['Q'] = leaf_node['V']
        for e in reversed(path):
            tree_env.tree[e[0]][e[1]]['N'] += 1
            current_node = tree_env.tree.nodes[e[0]]
            if self._algorithm == 'uct':
                current_node['V'] = (current_node['V'] * current_node['N'] +
                                     tree_env.tree.nodes[e[1]]['V']) / (current_node['N'] + 1)
            else:
                raise ValueError
            tree_env.tree[e[0]][e[1]]['Q'] = current_node['V']

        return tree_env.tree.nodes[0]['V']

    def _navigate(self, tree_env):
        state = tree_env.state
        tree_env.tree.nodes[state]['N'] += 1
        action = self._select(tree_env)
        next_state = tree_env.step(action)
        if next_state not in tree_env.leaves:
            return [[state, next_state]] + self._navigate(tree_env)
        else:
            return [[state, next_state]]

    def _select(self, tree_env):
        out_edges = [e for e in tree_env.tree.edges(tree_env.state)]
        if self._algorithm == 'uct':
            ucb_values = np.array([tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges], dtype=float)
            n_state_action = np.array([tree_env.tree[e[0]][e[1]]['N'] for e in out_edges])
            ucb_values += self._exploration_coeff * np.sqrt(
                np.log(tree_env.tree.nodes[tree_env.state]['N']) / (n_state_action + 1e-10)
            )
            max_a = np.random.choice(np.argwhere(ucb_values == np.max(ucb_values)).ravel())

            return max_a
        else:
            raise ValueError
