import pathlib
import pickle

import numpy as np
from joblib import Parallel, delayed

from mcts import MCTS
from tree_env import SyntheticTree


def experiment(algorithm, tree, alpha):
    mcts = MCTS(exploration_coeff=exploration_coeff,
                algorithm=algorithm,
                tau=tau,
                alpha=alpha,
                step_size=step_size,
                gamma=gamma,
                update_type='mean')

    v_hat, regret = mcts.run(tree, n_simulations)
    diff = np.abs(v_hat - tree.optimal_v_root)
    diff_uct = np.abs(v_hat - tree.max_mean)

    return diff, diff_uct, regret


n_exp = 5
n_trees = 5
n_simulations = 10000
# n_simulations = 1000
# ks = [2, 4, 6, 8, 10, 12, 14, 16]
# ds = [1, 2, 3, 4, 5]
ks = [2]
ds = [1, 2]

# ks = [8]
# ds = [5]

exploration_coeff = .1
tau = .1
alpha = .5
# alphas = [2,3,4] #power constant
gamma = 1.
step_size = 0.1
# algorithms = {'uct': 'UCT', 'tents': 'TENTS', 'w-mcts': 'W-MCTS', 'dng': 'DNG'}

algorithms = {'uct': 'UCT', 'ments': 'MENTS', 'rents': 'RENTS', 'tents': 'TENTS'}

# algorithms = {'power-uct-2': 'Power-UCT,p=2', 'power-uct-3': 'Power-UCT,p=3',
#               'power-uct-4': 'Power-UCT,p=4'}

# algorithms = {'alpha-divergence': 'ALPHA', 'tents': 'TENTS'}
# algorithms = {'w-mcts': 'W-MCTS'}

folder_name = './logs/expl_%.2f_tau_%.2f' % (exploration_coeff, tau)

diff_heatmap = np.zeros((len(algorithms), len(ks), len(ds)))
diff_uct_heatmap = np.zeros_like(diff_heatmap)
regret_heatmap = np.zeros_like(diff_heatmap)
for x, k in enumerate(ks):
    for y, d in enumerate(ds):
        subfolder_name = folder_name + '/k_' + str(k) + '_d_' + str(d)
        pathlib.Path(subfolder_name).mkdir(parents=True, exist_ok=True)
        for z, alg in enumerate(algorithms.keys()):
            print('Branching factor: %d, Depth: %d, Alg: %s' % (k, d, alg))
            out = list()
            for w in range(n_trees):
                try:
                    with open(subfolder_name + '/tree%d_%s.pkl' % (w, alg), 'rb') as f:
                        tree = pickle.load(f)
                except FileNotFoundError as err:
                    print('Tree not found! Creating new tree...')
                    tree = SyntheticTree(k, d, alg, tau,alpha,gamma,step_size)
                    with open(subfolder_name + '/tree%d_%s.pkl' % (w, alg), 'wb') as f:
                        pickle.dump(tree, f)

                out += Parallel(n_jobs=-1)(delayed(experiment)(alg, tree, alpha) for _ in range(n_exp))
            out = np.array(out)

            diff = out[:, 0]
            diff_uct = out[:, 1]
            regret = out[:, 2]

            avg_diff = diff.mean(0)
            avg_diff_uct = diff_uct.mean(0)
            avg_regret = regret.mean(0)
            diff_heatmap[z, x, y] = avg_diff[-1]
            diff_uct_heatmap[z, x, y] = avg_diff_uct[-1]
            regret_heatmap[z, x, y] = avg_regret[-1]

            np.save(subfolder_name + '/diff_%s.npy' % (alg), diff)
            np.save(subfolder_name + '/diff_uct_%s.npy' % (alg), diff_uct)
            np.save(subfolder_name + '/regret_%s.npy' % (alg), regret)

np.save(folder_name + '/diff_heatmap.npy', diff_heatmap)
np.save(folder_name + '/diff_uct_heatmap.npy', diff_uct_heatmap)
np.save(folder_name + '/regret_heatmap.npy', regret_heatmap)
