import numpy as np
from joblib import Parallel, delayed

from matplotlib import pyplot as plt
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

    return v_hat, diff, diff_uct


n_exp = 5
n_trees = 5
n_simulations = 10000
exploration_coeff = .1
ks = [5, 8, 10, 25, 50]
ds = [1, 2, 3, 4, 5]
tau = .01
algorithms = {'uct': 'UCT', 'ments': 'MENTS', 'rents': 'RENTS', 'tents': 'TENTS'}
plot = False

diff_heatmap = np.zeros((len(algorithms), len(ks), len(ds)))
diff_uct_heatmap = np.zeros((len(algorithms), len(ks), len(ds)))
for x, k in enumerate(ks):
    for y, d in enumerate(ds):
        for z, alg in enumerate(algorithms.keys()):
            print('Branching factor: %d, Depth: %d, Alg: %s' % (k, d, alg))
            out = Parallel(n_jobs=-1)(delayed(experiment)(alg, k, d) for _ in range(n_exp))
            v_hat = np.array([o[0]] for o in out)
            diff = np.array([o[1] for o in out])
            diff_uct = np.array([o[2] for o in out])

            avg_diff = diff.mean(0).mean(0)
            avg_diff_uct = diff_uct.mean(0).mean(0)
            diff_heatmap[z, x, y] = avg_diff[-1]
            diff_uct_heatmap[z, x, y] = avg_diff_uct[-1]

            if plot:
                plt.subplot(2, 1, 1)
                plt.plot(avg_diff)
                err = 2 * np.std(diff.reshape(n_exp * n_trees, n_simulations),
                                 axis=0) / np.sqrt(n_exp * n_trees)
                plt.fill_between(np.arange(n_simulations), avg_diff - err, avg_diff + err,
                                 alpha=.5)

                plt.subplot(2, 1, 2)
                plt.plot(avg_diff_uct)
                err = 2 * np.std(diff_uct.reshape(n_exp * n_trees, n_simulations),
                                 axis=0) / np.sqrt(n_exp * n_trees)
                plt.fill_between(np.arange(n_simulations), avg_diff_uct - err,
                                 avg_diff_uct + err, alpha=.5)

if plot:
    for i in range(1, 3):
        plt.subplot(2, 1, i)
        plt.ylim(0, 1)
        plt.grid()
        plt.legend(algorithms.values())
    plt.show()

np.save('diff_heatmap_expl_%f_tau_%f.npy' % (exploration_coeff, tau),
        diff_heatmap)
np.save('diff_uct_heatmap_%f_tau_%f.npy' % (exploration_coeff, tau),
        diff_uct_heatmap)
