"""random_survival_prob.py

Compute survival probability of point agent with budget B and easyness goal area/total area
"""

import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# ## for Palatino and other serif fonts use:
# #rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)

# loop over
# goal area / total area ratios
# budget
# space dimensions
Budget = 1000
size_space = 1.0
size_goal = 0.35

binom_n = Budget + 0
binom_k = 1
binom_p = size_goal/size_space

ps = []
# dims = [1, 2, 3, 4, 5, 6, 10, 20, 50, 100]
# dims = np.exp(np.linspace(0, 4, 20)).astype(int)
# dims = np.power(2, np.linspace(0, 5, 12)).astype(int)
dims = np.linspace(1, 7, 7).astype(int)
dim_colors = np.linspace(1, 7, 7).astype(int)
cmap = plt.get_cmap('tab20c')

print(dims)

for dim in dims:
    vol_space = size_space**dim
    vol_goal = (size_goal**dim) * vol_space # np.pi * 
    p = np.log(vol_goal/vol_space)
    print("p_%d = %f" % (dim, np.exp(p)))
    ps.append(p)

ps = np.array(ps)

# p_success = ps
# exp(p) is used a generalization of p(x^d) = goal-size / length-of-dimension^d
p_fail_dim = []
Budget_reqs = []
for i, dim in enumerate(dims):
    p = ps[i]
    Budget_req = Budget * (1-np.exp(p))
    print("Budget_req", Budget_req)
    Budget_reqs.append(Budget_req)
    # p_fail = np.random.binomial(n = binom_n, p = 1 - np.exp(p), size = 1000)
    p_fail = np.random.binomial(n = binom_n, p = np.exp(p), size = 1000)
    p_fail_dim.append(p_fail)

p_fail_dim = np.array(p_fail_dim)
Budget_reqs = np.array(Budget_reqs)

# print p_fail_dim.shape

# p_nohit 

# fig = plt.figure()

# plt.subplot(211)

row_num = 2
row_logp = 0
row_logp_span = 1
row_hists = 1

ax = plt.subplot2grid((row_num, len(dims)), (row_logp, 0), colspan=len(dims), rowspan=row_logp_span)
# ax.plot(dims, ps, "ko")
for i, dim in enumerate(dims):
    # ax.plot([dim], ps[i], color=cmap.colors[i], marker="o", markersize=10, alpha=0.7)
    ax.plot([dim], np.exp(ps[i]), color=cmap.colors[i], marker="o", markersize=10, alpha=0.7)
    ax.text(dim + 0.1, ps[i]-0.25, round(np.exp(ps[i]), 2))
# ax.set_title('Number of failures in episodes of budget-size = 1000')

# ax.plot(dims, (Budget_reqs - np.min(Budget_reqs))/np.max(Budget_reqs), "ro")
ax.set_xlabel('Number of state space dimensions d')
ax.set_ylabel('p')
# ax.set_ylabel('log(p)')
# ax.set_yscale('log')
ylim = ax.get_ylim()
ax.set_ylim((ylim[0] - 0.04, ylim[1] + 0.04))

# plt.subplot(212)
ax = plt.subplot2grid((row_num, len(dims)), (row_hists, 0), colspan=len(dims))
for i, dim in enumerate(dims):
    # ax = plt.subplot2grid((row_num, len(dims)), (row_hists, i), colspan=1)
    h, bins = np.histogram(p_fail_dim[i], bins = 'auto', density = False)
    # print "h", h, "bins", bins
    # h_ = h / np.sum(h)
    h_ = h.copy()
    bins_ = bins[:-1] + np.diff(bins)
    # plt.bar(left = bins_, height = h_, bottom = 0, width = 1.0)
    # ax.bar(x = bins_, height = h_, bottom = 0, width = 0.125 * max(5.0, (bins_[-1] - bins_[0])), color='black')
    ax.bar(x = bins_, height = h_, bottom = 0, width = 10.0, color=cmap.colors[i], alpha=0.5)
    # ax.plot([Budget_reqs[i]] * 2, [0, 1.0], "r-")
    # print ax.get_xlim()
    
    # xlim = np.array(ax.get_xlim())
    # if xlim[1] - xlim[0] < 10.0:
    #     xlim[1] += 1.
    #     xlim[0] = xlim[1] - 10
    # ax.set_xlim(xlim)
    # ax.set_xticks(xlim)
    # ax.set_xticklabels([int(xlim[0]), min(10000, int(xlim[1]))], fontsize=6)

    # ax.set_xlim((500 - 10, binom_n + 10))
    # ax.set_ylim((0, 0.4))
    # ax.set_yscale('log')
    
    # if i > (len(dims)-2):
    #     ax.set_yticklabels([])
    # plt.hist(p_fail_dim[i], bins = 'auto', normed = True)
    # plt.gca().set_yscale('log')
# plt.ylabel('log(p)')
ax.set_xlabel('Number of successes over 1000 episodes of 1000 steps each')
ax.set_ylabel('Success count')

hspace = 0.5
wspace = 0.3
        
fig = ax.figure
# fig.suptitle("Probability of success for random strategy with $ \\rho = {0} $".format(binom_p))
fig.suptitle("Probability of success for random strategy for p = {0}".format(binom_p))
fig.subplots_adjust(bottom=0.12, wspace=wspace, hspace=hspace)
fig.set_size_inches((12, 2.5*row_num))
fig.savefig('random_survival_prob_v4.pdf', dpi = 300, bbox_inches = None) # 'tight'
# fig.savefig('random_survival_prob.pdf', dpi = 300, bbox_inches = None)
    
plt.show()
