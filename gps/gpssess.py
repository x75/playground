import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# df = pd.read_csv('~/schwanderberg-gigerhorn-1.csv', delimiter=' ')
df = pd.read_csv('~/schwanderberg-niederental-clean.csv', delimiter=' ')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(df['lat'], df['lon'], df['elevation(m)'])
# plt.draw()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(df['lat'], df['lon'], df['elevation(m)'], 'k', linestyle='-', marker='.', alpha=0.5)
# ax.set_aspect(1)

plt.show()
