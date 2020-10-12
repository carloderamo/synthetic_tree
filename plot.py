import numpy as np
from matplotlib import pyplot as plt


n_exp = 5
n_trees = 5
k = ['2', '4', '6', '8', '10']
d = ['1', '2', '3', '4', '5']
n_simulations = 10
exploration_coeff = .5
tau = .01
algs = ['uct', 'ments', 'rents', 'tents']

folder_name = 'logs/2020-10-12_23-33-46'

diff = np.load(folder_name + '/diff_heatmap_expl_%.2f_tau_%.2f.npy' % (
    exploration_coeff, tau)
)
diff_uct = np.load(folder_name + '/diff_uct_heatmap_expl_%.2f_tau_%.2f.npy' % (
    exploration_coeff, tau)
)
diffs = [diff, diff_uct]
titles_diff = ['DIFF', 'DIFF_UCT']
for t, d in zip(titles_diff, diffs):
    plt.figure()
    plt.suptitle(t)
    for i, alg in enumerate(algs):
        plt.subplot(2, 2, i + 1)
        plt.imshow(d[i], cmap=plt.get_cmap('inferno'))
        plt.xticks(np.arange(5), d)
        plt.yticks(np.arange(5), k)
        plt.title(algs[i].upper())
        plt.colorbar()
plt.show()

subfolder_name = folder_name + '/k_%d_d_%d' % (k, d)
plt.figure()
for alg in algs:
    diff = np.load(folder_name + '/diff_%s_expl_%.2f_tau_%.2f.npy' % (
        alg, exploration_coeff, tau)
    )
    avg_diff = diff.mean(0).mean(0)
    plt.subplot(2, 1, 1)
    plt.plot(avg_diff)
    err = 2 * np.std(diff.reshape(n_exp * n_trees, n_simulations),
                     axis=0) / np.sqrt(n_exp * n_trees)
    plt.fill_between(np.arange(n_simulations), avg_diff - err, avg_diff + err,
                     alpha=.5)

    diff_uct = np.load(folder_name + '/diff_uct_%s_expl_%.2f_tau_%.2f.npy' % (
        alg, exploration_coeff, tau)
    )
    avg_diff_uct = diff.mean(0).mean(0)
    plt.subplot(2, 1, 2)
    plt.plot(avg_diff_uct)
    err = 2 * np.std(diff_uct.reshape(n_exp * n_trees, n_simulations),
                     axis=0) / np.sqrt(n_exp * n_trees)
    plt.fill_between(np.arange(n_simulations), avg_diff_uct - err,
                     avg_diff_uct + err, alpha=.5)

for i in range(1, 3):
    plt.subplot(2, 1, i)
    plt.grid()
    plt.legend([alg.upper() for alg in algs])
plt.show()
