import numpy as np
from scipy.special import logsumexp


class MCTS:
    def __init__(self, exploration_coeff, algorithm, tau):
        self._exploration_coeff = exploration_coeff
        self._algorithm = algorithm
        self._tau = tau

    def run(self, tree_env, n_simulations):
        v_hat = np.zeros(n_simulations)
        regret = np.zeros_like(v_hat)
        for i in range(n_simulations):
            tree_env.reset()
            v_hat[i], regret[i] = self._simulation(tree_env)

        return v_hat, regret.cumsum()

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
            else:
                out_edges = [e for e in tree_env.tree.edges(e[0])]
                qs = np.array(
                    [tree_env.tree[e[0]][e[1]]['Q'] for e in out_edges])
                if self._algorithm == 'ments':
                    current_node['V'] = self._tau * logsumexp(qs / self._tau)
                elif self._algorithm == 'rents':
                    qs_tau = qs / self._tau
                    weighted_logsumexp_qs = qs_tau.max() + np.log(
                        np.sum(current_node['prior'] * np.exp(qs_tau - qs_tau.max()))
                    )
                    current_node['V'] = self._tau * weighted_logsumexp_qs
                elif self._algorithm == 'tents':
                    q_tau = qs / self._tau
                    temp_q_tau = q_tau.copy()

                    sorted_q = np.flip(np.sort(temp_q_tau))
                    kappa = list()
                    for i in range(1, len(sorted_q) + 1):
                        if 1 + i * sorted_q[i-1] > sorted_q[:i].sum():
                            idx = np.argwhere(temp_q_tau == sorted_q[i-1]).ravel()[0]
                            temp_q_tau[idx] = np.nan
                            kappa.append(idx)
                    kappa = np.array(kappa)

                    sparse_max = q_tau[kappa] ** 2 / 2 - (q_tau[kappa].sum() - 1) ** 2 / (2 * len(kappa) ** 2)
                    sparse_max = sparse_max.sum() + .5
                    current_node['V'] = self._tau * sparse_max
                else:
                    raise ValueError

            current_node['N'] += 1

        v_hat = tree_env.tree.nodes[0]['V']
        max_a = self._select(tree_env=tree_env, state=0)
        regret = tree_env.q_root.max() - tree_env.q_root[max_a]

        return v_hat, regret

    def _navigate(self, tree_env):
        state = tree_env.state
        action = self._select(tree_env, state)
        next_state = tree_env.step(action)
        if next_state not in tree_env.leaves:
            return [[state, next_state]] + self._navigate(tree_env)
        else:
            return [[state, next_state]]

    def _select(self, tree_env, state):
        out_edges = [e for e in tree_env.tree.edges(state)]
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

            chosen_action = np.random.choice(np.argwhere(ucb_values == np.max(ucb_values)).ravel())
            probs = np.zeros_like(ucb_values)
            probs[chosen_action] += 1

            return chosen_action
        else:
            n_actions = len(out_edges)
            lambda_coeff = np.clip(self._exploration_coeff * n_actions / np.log(
                np.sum(n_state_action) + 1 + 1e-10), 0, 1)

            if self._algorithm == 'ments':
                q_exp_tau = np.exp(qs / self._tau)
                probs = (1 - lambda_coeff) * q_exp_tau / q_exp_tau.sum() + lambda_coeff / n_actions
            elif self._algorithm == 'rents':
                qs_tau = qs / self._tau
                prior_q_exp_tau = tree_env.tree.nodes[state]['prior'] * np.exp(qs_tau - qs_tau.max())
                probs = (1 - lambda_coeff) * prior_q_exp_tau / (prior_q_exp_tau.sum()) + lambda_coeff / n_actions
            elif self._algorithm == 'tents':
                q_tau = qs / self._tau
                temp_q_tau = q_tau.copy()

                sorted_q = np.flip(np.sort(temp_q_tau))
                kappa = list()
                for i in range(1, len(sorted_q) + 1):
                    if 1 + i * sorted_q[i-1] > sorted_q[:i].sum():
                        idx = np.argwhere(temp_q_tau == sorted_q[i-1]).ravel()[0]
                        temp_q_tau[idx] = np.nan
                        kappa.append(idx)
                kappa = np.array(kappa)

                max_omega = np.maximum(q_tau - (q_tau[kappa].sum() - 1) / len(kappa),
                                       np.zeros(len(q_tau)))
                probs = (1 - lambda_coeff) * max_omega + lambda_coeff / n_actions
            else:
                raise ValueError

            return np.random.choice(np.arange(n_actions), p=probs)
