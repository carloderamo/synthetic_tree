import numpy as np
from matplotlib import pyplot as plt


n_exp = 5
n_trees = 5
ks = [2, 4, 6, 8, 10, 12, 14, 16]
ds = [1, 2, 3, 4, 5]
k = 16
d = 5
n_simulations = 10000
exploration_coeff = .5
tau = .1
algs = ['uct', 'ments', 'rents', 'tents']

folder_name = 'results/expl_%.2f_tau_%.2f' % (exploration_coeff, tau)

# PLOTS
max_diff = 0
max_diff_uct = 0
max_regret = 0
plt.figure()
for alg in algs:
    subfolder_name = folder_name + '/k_%d_d_%d' % (k, d)
    diff = np.load(subfolder_name + '/diff_%s.npy' % (alg))
    avg_diff = diff.mean(0)
    plt.subplot(3, 1, 1)
    plt.plot(avg_diff)
    err = 2 * np.std(diff.reshape(n_exp * n_trees, n_simulations),
                     axis=0) / np.sqrt(n_exp * n_trees)
    plt.fill_between(np.arange(n_simulations), avg_diff - err, avg_diff + err,
                     alpha=.5)
    max_diff = max(max_diff, avg_diff.max())

    diff_uct = np.load(subfolder_name + '/diff_uct_%s.npy' % (alg))
    avg_diff_uct = diff_uct.mean(0)
    plt.subplot(3, 1, 2)
    plt.plot(avg_diff_uct)
    err = 2 * np.std(diff_uct.reshape(n_exp * n_trees, n_simulations),
                     axis=0) / np.sqrt(n_exp * n_trees)
    plt.fill_between(np.arange(n_simulations), avg_diff_uct - err,
                     avg_diff_uct + err, alpha=.5)
    max_diff_uct = max(max_diff_uct, avg_diff_uct.max())

    regret = np.load(subfolder_name + '/regret_%s.npy' % (alg))
    avg_regret = regret.mean(0)
    plt.subplot(3, 1, 3)
    plt.plot(avg_regret)
    err = 2 * np.std(regret.reshape(n_exp * n_trees, n_simulations),
                     axis=0) / np.sqrt(n_exp * n_trees)
    plt.fill_between(np.arange(n_simulations), avg_regret - err,
                     avg_regret + err, alpha=.5)
    max_regret = max(max_regret, avg_regret.max())

plots = [max_diff, max_diff_uct, max_regret]
for i in range(1, 4):
    plt.subplot(3, 1, i)
    plt.grid()
    plt.ylim(0, plots[i-1])
    plt.legend([alg.upper() for alg in algs])

# HEATMAPS
diff = np.load(folder_name + '/diff_heatmap.npy')
diff_uct = np.load(folder_name + '/diff_uct_heatmap.npy')
regret = np.load(folder_name + '/regret_heatmap.npy')

diffs = [diff, diff_uct, regret]
titles_diff = ['DIFF', 'DIFF_UCT', 'PSEUDO_REGRET']
for t, d in zip(titles_diff, diffs):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(t)
    max_d = d.max()
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(d[i], cmap=plt.get_cmap('inferno'))
        ax.set_title(algs[i].upper())
        ax.set_xticks(np.arange(len(ds)))
        ax.set_yticks(np.arange(len(ks)))
        ax.set_xticklabels(ds)
        ax.set_yticklabels(ks)
        im.set_clim(0, max_d)
    cb_ax = fig.add_axes([0.875, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cb_ax)

plt.show()
