import pathlib
import pickle

import numpy as np
from joblib import Parallel, delayed

from mcts import MCTS
from tree_env import SyntheticTree


def experiment(algorithm, tree, epsilon):
    mcts = MCTS(exploration_coeff=epsilon,
                algorithm=algorithm,
                tau=tau)

    v_hat, regret = mcts.run(tree, n_simulations)
    diff = np.abs(v_hat - tree.optimal_v_root)
    diff_uct = np.abs(v_hat - tree.max_mean)

    return diff, diff_uct, regret


n_exp = 5
n_trees = 5
n_simulations = 10000
k = 8
d = 3
epsilons = [.01, .025, .05, .075, .1, .25, .5, .75, 1.]
taus = [.01, .025, .05, .075, .1, .25, .5, .75, 1.]
algorithms = {'uct': 'UCT', 'ments': 'MENTS', 'rents': 'RENTS', 'tents': 'TENTS'}

folder_name = './logs/k_%d_d_%d' % (k, d)

diff_heatmap = np.zeros((len(algorithms), len(epsilons), len(taus)))
diff_uct_heatmap = np.zeros_like(diff_heatmap)
regret_heatmap = np.zeros_like(diff_heatmap)
for x, eps in enumerate(epsilons):
    for y, tau in enumerate(taus):
        subfolder_name = folder_name + '/eps_%.3f_tau_%.3f' % (eps, tau)
        pathlib.Path(subfolder_name).mkdir(parents=True, exist_ok=True)
        for z, alg in enumerate(algorithms.keys()):
            print('Epsilon: %.3f, Tau: %.3f, Alg: %s' % (eps, tau, alg))
            out = list()
            for w in range(n_trees):
                try:
                    with open(subfolder_name + '/tree%d_%s.pkl' % (w, alg), 'rb') as f:
                        tree = pickle.load(f)
                except FileNotFoundError as err:
                    print('Tree not found! Creating new tree...')
                    tree = SyntheticTree(k, d, alg, tau)
                    with open(subfolder_name + '/tree%d_%s.pkl' % (w, alg), 'wb') as f:
                        pickle.dump(tree, f)

                out += Parallel(n_jobs=-1)(delayed(experiment)(alg, tree, eps) for _ in range(n_exp))
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
