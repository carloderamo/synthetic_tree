import numpy as np
from scipy.special import logsumexp


class MCTS:
    def __init__(self, exploration_coeff, algorithm, tau):
        self._exploration_coeff = exploration_coeff
        self._algorithm = algorithm
        self._tau = tau

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
        leaf_node['N'] += 1
        for e in reversed(path):
            current_node = tree_env.tree.nodes[e[0]]
            next_node = tree_env.tree.nodes[e[1]]

            tree_env.tree[e[0]][e[1]]['Q'] = next_node['V']
            tree_env.tree[e[0]][e[1]]['N'] += 1
            if self._algorithm == 'uct':
                current_node['V'] = (current_node['V'] * current_node['N'] +
                                     tree_env.tree[e[0]][e[1]]['Q']) / (current_node['N'] + 1)
            elif self._algorithm == 'ments':
                out_edges = [e for e in tree_env.tree.edges(e[0])]
                qs = np.array(
                    [tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])
                current_node['V'] = self._tau * logsumexp(qs / self._tau)
            elif self._algorithm == 'rents':
                out_edges = [e for e in tree_env.tree.edges(e[0])]
                qs = np.array(
                    [tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])
                visitation_ratio = np.array(
                    [tree_env.tree[e[0]][e[1]]['N'] / (tree_env.tree.nodes[e[0]][
                        'N'] + 1e-10) for e in out_edges]
                )
                current_node['V'] = self._tau * np.log(np.sum(visitation_ratio * np.exp(qs / self._tau)))
            else:
                raise ValueError

            current_node['N'] += 1

        return tree_env.tree.nodes[0]['V']

    def _navigate(self, tree_env):
        state = tree_env.state
        action = self._select(tree_env)
        next_state = tree_env.step(action)
        if next_state not in tree_env.leaves:
            return [[state, next_state]] + self._navigate(tree_env)
        else:
            return [[state, next_state]]

    def _select(self, tree_env):
        out_edges = [e for e in tree_env.tree.edges(tree_env.state)]
        n_state_action = np.array(
            [tree_env.tree[e[0]][e[1]]['N'] for e in out_edges])
        qs = np.array(
            [tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])
        if self._algorithm == 'uct':
            n_state = np.sum(n_state_action)
            if n_state > 0:
                ucb_values = qs + self._exploration_coeff * np.sqrt(
                    np.log(n_state) / (n_state_action + 1e-10)
                )
            else:
                ucb_values = np.ones(len(n_state_action)) * np.inf

            return np.random.choice(np.argwhere(ucb_values == np.max(ucb_values)).ravel())
        elif self._algorithm == 'ments':
            n_actions = len(out_edges)
            lambda_coeff = np.clip(self._exploration_coeff * n_actions / np.log(
                np.sum(n_state_action) + 1 + 1e-10), 0, 1)
            q_exp_tau = np.exp(qs / self._tau)
            probs = (1 - lambda_coeff) * q_exp_tau / q_exp_tau.sum() + lambda_coeff / n_actions

            return np.random.choice(np.arange(n_actions), p=probs)
        elif self._algorithm == 'rents':
            n_actions = len(out_edges)
            lambda_coeff = np.clip(self._exploration_coeff * n_actions / np.log(
                np.sum(n_state_action) + 1 + 1e-10), 0, 1)
            q_exp_tau = np.exp(qs / self._tau)
            visitation_ratio = np.array(
                [tree_env.tree[e[0]][e[1]]['N'] / (tree_env.tree.nodes[e[0]]['N'] + 1e-10) for e in out_edges]
            )
            probs = (1 - lambda_coeff) * visitation_ratio * q_exp_tau / q_exp_tau.sum() + lambda_coeff / n_actions
            probs[np.random.randint(len(probs))] += 1 - probs.sum()

            return np.random.choice(np.arange(n_actions), p=probs)
        else:
            raise ValueError
