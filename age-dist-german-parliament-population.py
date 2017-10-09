# coding: utf-8

data = np.genfromtxt('/home/lib/projects/zerotrust/bundestag-WP18.csv', delimiter=',', skip_header=1)
df = pd.read_csv('/home/lib/projects/zerotrust/bundestag-WP18.csv', sep=',')
plt.ion()
df.plot()
plt.hist(data[:,1])
plt.hist(data[:,1])
get_ipython().magic(u'pinfo plt.hist')
plt.hist(data[:,1], rwidth=0.5)
plt.hist(2017 - data[:,1], rwidth=0.5)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticks([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6,)
         52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9,
         76.6,  79.3,  82. ]
plt.gca().set_xticks([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6,)
         52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9,
         76.6,  79.3,  82. ])
plt.gca().set_xticks([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticks([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])
plt.gca().set_xtickmarks([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])
plt.gca().set_xtickmark([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])
plt.gca().set_xticklabels([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticks(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])))
plt.gca().set_xticklabels(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])))
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticklabels(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])) + 28.0)
plt.gca().set_xticks(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ]) + 28.0))
plt.gca().set_xticklabels(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ]) + 28.0))
plt.gca().set_xticklabels(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])) + 28.0)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticks(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])) + 28.0)
plt.gca().set_xticklabels(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ])) + 28.0)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticks(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ]) + 29.35)
plt.gca().set_xticklabels(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ]) + 29.35)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticks(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ]) + 28.0)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticks(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ]))
plt.gca().set_xticklabels(np.cumsum(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ]))) + 28.0)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.gca().set_xticklabels(np.cumsum(np.diff(np.array([ 28. ,  30.7,  33.4,  36.1,  38.8,  41.5,  44.2,  46.9,  49.6, 52.3,  55. ,  57.7,  60.4,  63.1,  65.8,  68.5,  71.2,  73.9, 76.6,  79.3,  82. ]))) + 28.0)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
get_ipython().magic(u'pinfo plt.hist')
plt.hist(2017 - data[:,1], bins = 20, rwidth='none')
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
plt.hist(2017 - data[:,1], bins = 20, rwidth=0.8)
np.min(data[:,1])
np.max(data[:,1])
1989-1935
plt.hist(2017 - data[:,1], bins = 54, rwidth=0.8)
plt.hist(2017 - data[:,1], bins = 54, rwidth=0.8)
plt.suptitle('age histogram of german parliament members')
plt.hist(2017 - data[:,1], bins = 54, rwidth=0.8)
plt.suptitle('age histogram of german parliament members')
plt.gca().set_xlim((0, 100))
df
df[:,Partei='SPD']
df[Partei='SPD']
df
df
df[Partei]
df[Partei='SPD']
df[...,Partei='SPD']
get_ipython().magic(u'pinfo df')
df[0]
df[1]
df = pd.read_csv('/home/lib/projects/zerotrust/bundestag-WP18.csv', sep=',')
df[1]
df[:]
df[:,0]
df[:][0]
df[:][Partei='SPD']
get_ipython().magic(u'pinfo pd.DataFrame')
pd.Index
get_ipython().magic(u'pinfo pd.Index')
df[col='Partei']
df['Partei']
df['Partei' == 'SPD']
df[Partei == 'SPD']
df['Partei' = 'SPD']
df['Partei']
df['Partei'][:10]
df['Partei']
df['Partei'] == 'SPD'
df[...,df['Partei'] == 'SPD']
df[df['Partei'] == 'SPD']
df[df['Partei'] == 'SPD',1]
df[df['Partei'] == 'SPD',[1]]
df[df['Partei'] == 'SPD']
df[df['Partei'] == 'SPD'][1]
df[df['Partei'] == 'SPD']
df[df['Partei'] == 'SPD',0]
df[df['Partei'] == 'SPD',1]
df[df['Partei'] == 'SPD'].shape
df[df['Partei'] == 'SPD'][0]
df[df['Partei'] == 'SPD'][1]
type(df[df['Partei'] == 'SPD'])
df[df['Partei'] == 'SPD']
df[df['Partei'] == 'SPD'][:]
df[df['Partei'] == 'SPD'][:,1]
df[df['Partei'] == 'SPD'][:,0]
df[df['Partei'] == 'SPD'][:,[0]]
df[df['Partei'] == 'SPD'][:]
df[df['Partei'] == 'SPD']
df[df['Partei'] == 'SPD','Partei']
df[df['Partei'] == 'SPD']['Partei']
df[df['Partei'] == 'SPD']['geb.']
plt.plot(df[df['Partei'] == 'SPD']['geb.'])
plt.hist(df[df['Partei'] == 'SPD']['geb.'])
plt.hist(2017 - df[df['Partei'] == 'SPD']['geb.'])
plt.hist(2017 - df[df['Partei'] == 'CSU']['geb.'])
plt.hist(2017 - df[df['Partei'] == 'CDU']['geb.'])
df
df.columns
df.columns()
df.index
df.names
df.names()
df.columns
plt.hist(2017 - df[df['Partei'] == 'CDU']['Partei'])
plt.hist(2017 - df[df['Partei'] == 'CDU']['Partei'])
df.columns
plt.hist(2017 - df[df['Partei'] == 'CDU']['geb.'])
df[df['Partei'] == 'CDU']['Partei']
df['Partei']
df['Partei'].uniquie()
df['Partei'].unique()
df['Partei'].unique()
plt.hist(2017 - df[df['Partei'] == 'DIE LINKE']['geb.'])
plt.hist(2017 - df[df['Partei'] == 'SPD']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'CSU']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'CDU']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'GR\xc3\x9cNE']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'fraktionslos']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'fraktionslos']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'GR\xc3\x9cNE']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'fraktionslos']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'CDU']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'CSU']['geb.'], alpha = 0.3)
plt.hist(2017 - df[df['Partei'] == 'SPD']['geb.'], alpha = 0.3)
get_ipython().magic(u'save age-dist-german-parliament-population')
get_ipython().magic(u'save age-dist-german-parliament-population 0-141')
