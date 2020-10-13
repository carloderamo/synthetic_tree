import numpy as np
from matplotlib import pyplot as plt


n_exp = 5
n_trees = 5
ks = [2, 4, 6, 8, 10]
ds = [1, 2, 3, 4, 5]
k = 6
d = 3
n_simulations = 10000
exploration_coeff = .5
tau = .1
algs = ['uct', 'ments', 'rents', 'tents']

folder_name = 'logs/2020-10-12_23-57-17'

# PLOTS
subfolder_name = folder_name + '/k_%d_d_%d' % (k, d)
plt.figure()
for alg in algs:
    diff = np.load(subfolder_name + '/diff_%s_expl_%.2f_tau_%.2f.npy' % (
        alg, exploration_coeff, tau)
    )
    avg_diff = diff.mean(0).mean(0)
    plt.subplot(2, 1, 1)
    plt.plot(avg_diff)
    err = 2 * np.std(diff.reshape(n_exp * n_trees, n_simulations),
                     axis=0) / np.sqrt(n_exp * n_trees)
    plt.fill_between(np.arange(n_simulations), avg_diff - err, avg_diff + err,
                     alpha=.5)

    diff_uct = np.load(subfolder_name + '/diff_uct_%s_expl_%.2f_tau_%.2f.npy' % (
        alg, exploration_coeff, tau)
    )
    avg_diff_uct = diff_uct.mean(0).mean(0)
    plt.subplot(2, 1, 2)
    plt.plot(avg_diff_uct)
    err = 2 * np.std(diff_uct.reshape(n_exp * n_trees, n_simulations),
                     axis=0) / np.sqrt(n_exp * n_trees)
    plt.fill_between(np.arange(n_simulations), avg_diff_uct - err,
                     avg_diff_uct + err, alpha=.5)

for i in range(1, 3):
    plt.subplot(2, 1, i)
    plt.grid()
    plt.ylim(0, .5)
    plt.legend([alg.upper() for alg in algs])

# HEATMAPS
diff = np.load(folder_name + '/diff_heatmap_expl_%.2f_tau_%.2f.npy' % (
    exploration_coeff, tau)
)
diff_uct = np.load(folder_name + '/diff_uct_heatmap_expl_%.2f_tau_%.2f.npy' % (
    exploration_coeff, tau)
)
diffs = [diff, diff_uct]
titles_diff = ['DIFF', 'DIFF_UCT']
for t, d in zip(titles_diff, diffs):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(t)
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(d[i], cmap=plt.get_cmap('inferno'))
        ax.set_title(algs[i].upper())
        ax.set_xticks(np.arange(len(ds)))
        ax.set_yticks(np.arange(len(ks)))
        ax.set_xticklabels(ds)
        ax.set_yticklabels(ks)
        im.set_clim(0, .3)
    cb_ax = fig.add_axes([0.875, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cb_ax)

plt.show()
