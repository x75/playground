"""random_survival_prob.py

Compute survival probability of point agent with budget B and easyness goal area/total area
"""

import numpy as np
import matplotlib.pyplot as plt

# loop over
# goal area / total area ratios
# budget
# space dimensions
B = 1000
size_space = 1.0
size_goal = 0.1

ps = []
# dims = [1, 2, 3, 4, 5, 6, 10, 20, 50, 100]
# dims = np.exp(np.linspace(0, 4, 20)).astype(int)
# dims = np.power(2, np.linspace(0, 5, 12)).astype(int)
dims = np.linspace(1, 7, 7).astype(int)

print dims

for dim in dims:
    vol_space = size_space**dim
    vol_goal = (size_goal**dim) * vol_space # np.pi * 
    p = np.log(vol_goal/vol_space)
    print "p_%d = %f" % (dim, np.exp(p))
    ps.append(p)

ps = np.array(ps)

# p_success = ps
# exp(p) is used a generalization of p(x^d) = goal-size / length-of-dimension^d
p_fail_dim = []
B_reqs = []
for i, dim in enumerate(dims):
    p = ps[i]
    B_req = B * (1-np.exp(p))
    print "B_req", B_req
    B_reqs.append(B_req)
    p_fail = np.random.binomial(n = B, p = 1 - np.exp(p), size = B)
    p_fail_dim.append(p_fail)

p_fail_dim = np.array(p_fail_dim)
B_reqs = np.array(B_reqs)

# print p_fail_dim.shape

# p_nohit 

# fig = plt.figure()

# plt.subplot(211)
ax = plt.subplot2grid((3, len(dims)), (0, 0), colspan=len(dims), rowspan = 2)
ax.plot(dims, ps, "ko")
# ax.plot(dims, (B_reqs - np.min(B_reqs))/np.max(B_reqs), "ro")
ax.set_ylabel('log(p)')
# plt.subplot(212)
for i, dim in enumerate(dims):
    ax = plt.subplot2grid((3, len(dims)), (2, i), colspan=1)
    h, bins = np.histogram(p_fail_dim[i], bins = 'auto', density = True)
    # print "h", h, "bins", bins
    h_ = h / np.sum(h)
    bins_ = bins[:-1] + np.diff(bins)
    # plt.bar(left = bins_, height = h_, bottom = 0, width = 1.0)
    ax.bar(left = bins_, height = h_, bottom = 0, width = 0.1 * max(5.0, (bins_[-1] - bins_[0])))
    ax.plot([B_reqs[i]] * 2, [0, 1.0], "r-")
    # print ax.get_xlim()
    xlim = np.array(ax.get_xlim())
    if xlim[1] - xlim[0] < 10.0:
        xlim[1] += 1.
        xlim[0] = xlim[1] - 10
    ax.set_xlim(xlim)
    ax.set_xticks(xlim)
    ax.set_xticklabels([int(xlim[0]), min(10000, int(xlim[1]))], fontsize=6)
    if i > 0:
        ax.set_yticklabels([])
    # plt.hist(p_fail_dim[i], bins = 'auto', normed = True)
    # plt.gca().set_yscale('log')
# plt.ylabel('log(p)')

fig = ax.figure
fig.suptitle("Probability of success for random strategy")
fig.set_size_inches((12, 5))
fig.savefig('random_survival_prob.pdf', dpi = 300, bbox_inches = 'tight')
    
plt.show()
