import datetime
import pathlib

import numpy as np
from joblib import Parallel, delayed

from mcts import MCTS
from tree_env import SyntheticTree


def experiment(algorithm, k, d):
    mcts = MCTS(exploration_coeff=exploration_coeff,
                algorithm=algorithm,
                tau=tau)

    v_hat = np.zeros((n_trees, n_simulations))
    diff = np.zeros_like(v_hat)
    diff_uct = np.zeros_like(v_hat)
    for i in range(n_trees):
        tree = SyntheticTree(k, d, algorithm, tau)
        v_hat[i] = mcts.run(tree, n_simulations)
        diff[i] = np.abs(v_hat[i] - tree.optimal_v_root)
        diff_uct[i] = np.abs(v_hat[i] - tree.max_mean)

    return diff, diff_uct


n_exp = 5
n_trees = 5
n_simulations = 10000
ks = [2, 4, 6, 8, 10]
ds = [1, 2, 3, 4, 5]
exploration_coeff = .5
tau = .01
algorithms = {'uct': 'UCT', 'ments': 'MENTS', 'rents': 'RENTS', 'tents': 'TENTS'}

diff_heatmap = np.zeros((len(algorithms), len(ks), len(ds)))
diff_uct_heatmap = np.zeros((len(algorithms), len(ks), len(ds)))
for x, k in enumerate(ks):
    for y, d in enumerate(ds):
        folder_name = './logs/' + 'k_' + str(k) + '_d_' + str(d)
        pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
        for z, alg in enumerate(algorithms.keys()):
            print('Branching factor: %d, Depth: %d, Alg: %s' % (k, d, alg))
            out = Parallel(n_jobs=-1)(delayed(experiment)(alg, k, d) for _ in range(n_exp))
            diff = np.array([o[0] for o in out])
            diff_uct = np.array([o[1] for o in out])

            avg_diff = diff.mean(0).mean(0)
            avg_diff_uct = diff_uct.mean(0).mean(0)
            diff_heatmap[z, x, y] = avg_diff[-1]
            diff_uct_heatmap[z, x, y] = avg_diff_uct[-1]
            
            np.save(folder_name + '/diff_%s_expl_%.2f_tau_%.2f.npy' % (
                alg, exploration_coeff, tau), diff
            )
            np.save(folder_name + '/diff_uct_%s_expl_%.2f_tau_%.2f.npy' % (
                alg, exploration_coeff, tau), diff_uct
            )

np.save(folder_name + '/diff_heatmap_expl_%.2f_tau_%.2f.npy' % (
    exploration_coeff, tau), diff_heatmap
)
np.save(folder_name + '/diff_uct_heatmap_%.2f_tau_%.2f.npy' % (
    exploration_coeff, tau), diff_uct_heatmap
)
