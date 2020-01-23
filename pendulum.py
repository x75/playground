"""Quick test of data asymmetry in a motor driven pendulum"""

import numpy as np
import matplotlib

import matplotlib.gridspec as gridspec

# matplotlib.use("Cairo")
import matplotlib.pyplot as plt


l = 2e-0
g = 9.81
m = 1.0
cf = -4.8 
# cf = -1.45 
fr = 1e-2
one_over_fr = 1 - fr

scale = 1.0 # 0.1

# dt = 0.5e-2
dt = 1e-2
# dt = 1e-1
numsteps = 10000

x = np.zeros((numsteps, 3))
# x[0,0] = -np.pi/2.0
x[0,0] = np.pi/2.1

g *= scale
cf *= scale

for i in range(1, numsteps):
    x[i,2] = (-g * m) / l * np.sin(x[i-1,0]) + cf + np.random.normal(0.0, 5e-0)
    x[i,1] = one_over_fr * x[i-1,1] + x[i,2] * dt
    x[i,0] = x[i-1,0] + x[i,1] * dt
    if x[i,0] > (1*np.pi):
        print("larger", x)
        x[i,0] = x[i,0] - (2 * np.pi)
    elif x[i,0] < (-1*np.pi):
        print("smaller", x)
        x[i,0] = x[i,0] + (2 * np.pi)

gs = gridspec.GridSpec(2, 3)
        
fig = plt.figure()
ax1 = fig.add_subplot(gs[0,:2])
# ax1.plot(np.abs(x))
# ax1.plot(x, "k,")
ax1.plot(x, "k-")
ax2 = fig.add_subplot(gs[0,2])
# si = x[:,0] ==
sls = 2000 # 1980+130
print("sls", sls)
# sls = x[sls:,0] == np.pi * 0.4
print("x[sls:,0]", x[sls:,0].shape)
print("x[sls:,0] < (np.pi * 0.4)", (x[sls:,0] < (np.pi * 0.4)))
print("x[sls:,0] > (np.pi * 0.39)", (x[sls:,0] > (np.pi * 0.39)))
sls += np.argwhere(np.logical_and(x[sls:,0] > (np.pi * 0.40), x[sls:,0] < (np.pi * 0.41)))[0,0]
sloffs = 10
sle  = sloffs + np.argwhere(np.logical_and(x[sls+sloffs:,0] > (np.pi * 0.40), x[sls+sloffs:,0] < (np.pi * 0.41)))[0,0]
print("sle", sle)
print("x[sls+sloffs+sle:,0]", x[sls+sloffs+sle:,0])
# sle  = np.argwhere(np.logical_and(x[sls:,0] > (np.pi * 0.6), x[sls:,0] < (np.pi * 0.61)))[0,0]
print("sls", sls)
sll = sle # int(115*6.8)
sl = slice(sls, sls + sll)
# ax2.plot([0, np.cos(x[sl,0]) * l], [0, np.sin(x[sl,0]) * l], "k-", alpha=0.1)
for x_ in x[sl,0]:
    ax2.plot([0, np.cos(x_) * l * 2], [0, np.sin(x_) * l * 2], "k-", linewidth=10.0, alpha=0.3)
ax2.plot(np.cos(x[sl,0]) * l * 2.05, np.sin(x[sl,0]) * l * 2.05, "ko", alpha=0.5)
ax2.set_aspect(1)

ax3 = fig.add_subplot(gs[1,:])
# print "x[:,0].shape", x[:,0].shape
# x_ = np.vstack((x[1:-1,0], x[2:,0])).T
rng = 1400 # detect freq of oscillation
x_ = np.vstack([x[i:-rng+i,0] for i in range(rng)]).T
print("x_", x_.shape)
# ax3.plot(x_[:,0], x_[:,1], "ko", alpha=0.25)
transoffs = 1000
x_comp = []
x_offs = []
y_comp = []
y_offs = []
for i in range(1, rng, 10):
    # mp = ax3.scatter(x_[:,0], x_[:,i], c = x[:-rng,1], alpha=0.25)
    # mp = ax3.scatter(x_[:,0], np.abs(x_[:,i]), c = x[:-rng,1], cmap=plt.get_cmap("gray"), alpha=float(i)/rng)
    sl = slice(transoffs, transoffs + 1000)
    x_comp.append(np.sin(x_[sl,0]))
    x_offs.append(np.ones_like(x_comp[-1]) * (i % (rng/10)))
    y_comp.append(np.sin(x_[sl,i]))
    y_offs.append(np.ones_like(y_comp[-1]) * 2.5 * (i // (rng/10)))
    # mp = ax3.scatter(x_comp + i, y_comp, c = x_comp - y_comp, cmap=plt.get_cmap("spectral"), alpha=float(i)/rng)
    # ax3.set_yscale("log")
x_comp = np.array(x_comp).flatten()
x_offs = np.array(x_offs).flatten()
y_comp = np.array(y_comp).flatten()
y_offs = np.array(y_offs).flatten()
mp = ax3.scatter(x_comp + x_offs, y_comp + y_offs, c = x_comp - y_comp, cmap=plt.get_cmap("Spectral"), alpha=0.25, s=2, linewidths=0)
    
# mp = ax3.scatter(x_[:,0], x_[:,1], c = x[:-1,1], alpha=0.25)
fig.colorbar(mappable = mp, ax = ax3, orientation="horizontal", fraction=0.05)
# ax3.set_aspect(1)

plt.show()
