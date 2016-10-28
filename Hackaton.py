
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic('pylab inline')
import datetime
import numpy as np
import matplotlib.pyplot as plt
try:
     from matplotlib.finance import quotes_historical_yahoo_ochl
except ImportError:
     # quotes_historical_yahoo_ochl was named quotes_historical_yahoo before matplotlib 1.4
    from matplotlib.finance import quotes_historical_yahoo as quotes_historical_yahoo_ochl
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold


# In[2]:

commodities = pd.read_csv('Commodities_data.csv')


# In[3]:

commoditiesmeta = pd.read_csv('Commodities_metadata.csv')


# In[4]:

commodities


# In[5]:

commodities_stocks = np.unique(commodities['Ticker'])


# In[6]:

commodities_stocks1 = ['CC1_COMB_Comdty', 'CL1_COMB_Comdty',
       'C_1_COMB_Comdty', 'HG1_COMB_Comdty', 'JO1_Comdty', 'KC2_Comdty',
       'LA1_Comdty', 'LH1_COMB_Comdty', 'LN1_Comdty', 'O_1_COMB_Comdty',
       'RR1_COMB_Comdty', 'SB1_Comdty', 'S_1_COMB_Comdty',
       'W_1_COMB_Comdty']


# In[7]:

commoditiesmeta['Commodity'].values


# In[8]:

commodities_meta = ['Corn', 'Cocoa', 'Oil', 'Copper', 'Orange Juice',
       'Coffee', 'Aluminum', 'Lean Pork', 'Nickel', 'Oats', 'Rice',
       'Soybeans', 'Sugar', 'Wheat']


# In[9]:

commoditiesmeta


# In[10]:

commodities_inst = {'BUT1_Comdty': [], 'CC1_COMB_Comdty' :[], 'CL1_COMB_Comdty': [],
       'C_1_COMB_Comdty':[], 'HG1_COMB_Comdty': [], 'JO1_Comdty' : [], 'KC2_Comdty' : [],
       'LA1_Comdty' : [], 'LH1_COMB_Comdty': [], 'LN1_Comdty' :[], 'O_1_COMB_Comdty':[],
       'RR1_COMB_Comdty':[], 'SB1_Comdty':[], 'S_1_COMB_Comdty':[],
      'W_1_COMB_Comdty':[] }
for i in commodities_stocks: 
    commodities_inst[i] = commodities[commodities['Ticker']==i]


# In[11]:

time = commodities_inst[commodities_stocks1[0]]['Date'].values
for i in range(1,len(commodities_stocks1)):
    time = np.intersect1d(time,commodities_inst[commodities_stocks1[i]]['Date'].values)
    print(time.shape)


# In[12]:

time


# In[13]:

for i in commodities_stocks1:
    print(commodities_inst[i])


# In[14]:

for i in commodities_stocks1:
    commodities_inst[i] = commodities_inst[i][commodities_inst[i]['Date'].isin(time)]


# In[15]:

for i in commodities_stocks1:    
    print(i)
    print(commodities_inst[i].shape)


# In[16]:

for i in commodities_stocks1:
    commodities_inst[i].reset_index(inplace=True)


# In[17]:

for i in commodities_stocks1:
    print(commodities_inst[i]['Low'].isnull().sum())


# In[18]:

for i in commodities_stocks1:
    if(commodities_inst[i]['Low'].isnull().sum()!=0):
        index = commodities_inst[i][commodities_inst[i]['Low'].isnull()==True].index


# In[19]:

index


# In[22]:

for i in commodities_stocks1:
    commodities_inst[i] = commodities_inst[i].drop(commodities_inst[i].index[index])


# In[23]:

for i in commodities_stocks1:
    commodities_inst[i]['Low_inc'] = np.zeros(commodities_inst[i].shape[0])
    commodities_inst[i]['Low_inc'][1:] = (commodities_inst[i]['Low'][1:].values-commodities_inst[i]['Low'][0:-1].values)/(commodities_inst[i]['Low'][0:-1].values)


# In[24]:

corr_comm = np.eye(len(commodities_stocks1))
for i in range(len(commodities_stocks1)-1):
    for j in range(i+1,len(commodities_stocks1)):
        corr_comm[i][j] = commodities_inst[commodities_stocks1[i]]['Low_inc'].corr(commodities_inst[commodities_stocks1[j]]['Low_inc'], method='pearson')


# In[25]:

corr_comm


# In[26]:

for i in range(len(commodities_stocks1)):
    col = commodities_inst[commodities_stocks1[i]]['Low_inc'].values[1:]
    print(col.shape)
    if(i == 0):
        result =  col
    else:
        result = np.vstack((result, col))


# In[125]:

prediction(2000,3,0)


# In[ ]:




# In[122]:

commodities['BUT1_Comdty']


# In[115]:

commodities_meta


# In[27]:

result.shape


# In[28]:

result[0]


# In[29]:

np.isnan(np.sum(result))


# In[30]:

edge_model = covariance.GraphLassoCV()
X = result.copy().T
X /= X.std(axis=0)
edge_model.fit(X)


# In[31]:

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

commodities_meta = np.asarray(commodities_meta)

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(commodities_meta[labels == i])))


# In[32]:

node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T


# In[33]:

plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(commodities_meta, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.savefig('Commodities.png')

plt.show()


# In[46]:

currencies = pd.read_csv('Currencies_data.csv')


# In[47]:

currencies_stocks = np.unique(currencies['Ticker'])


# In[48]:

currenciesmeta = pd.read_csv('Currencies_metadata.csv')


# In[49]:

currencies_stocks


# In[50]:

currencies


# In[51]:

currenciesmeta


# In[52]:

currencies_meta = currenciesmeta['SECURITY_NAME'].values


# In[53]:

currencies_inst = {'USDAED_Curncy':[], 'USDARS_Curncy':[], 'USDAUD_Curncy':[], 'USDBHD_Curncy':[],
       'USDBRL_Curncy':[], 'USDCAD_Curncy':[], 'USDCHF_Curncy':[], 'USDCLP_Curncy':[],
       'USDCNY_Curncy':[], 'USDCOP_Curncy':[], 'USDCZK_Curncy':[], 'USDDKK_Curncy':[],
       'USDEGP_Curncy':[], 'USDEUR_Curncy':[], 'USDGBP_Curncy':[], 'USDHKD_Curncy':[],
       'USDHUF_Curncy':[], 'USDIDR_Curncy':[], 'USDILS_Curncy':[], 'USDJOD_Curncy':[],
       'USDJPY_Curncy':[], 'USDKES_Curncy':[], 'USDKRW_Curncy':[], 'USDMAD_Curncy':[],
       'USDMXN_Curncy':[], 'USDMYR_Curncy':[], 'USDNGN_Curncy':[], 'USDNOK_Curncy':[],
       'USDNZD_Curncy':[], 'USDOMR_Curncy':[], 'USDPHP_Curncy':[], 'USDPKR_Curncy':[],
       'USDPLN_Curncy':[], 'USDQAR_Curncy':[], 'USDRUB_Curncy':[], 'USDSAR_Curncy':[],
       'USDSEK_Curncy':[], 'USDSGD_Curncy':[], 'USDTHB_Curncy':[], 'USDTRY_Curncy':[],
       'USDTWD_Curncy':[], 'USDUAH_Curncy':[], 'USDUGX_Curncy':[], 'USDVEF_Curncy':[],
       'USDVND_Curncy':[], 'USDZAR_Curncy':[] }
for i in currencies_stocks: 
    currencies_inst[i] = currencies[currencies['Ticker']==i]


# In[54]:

time = currencies_inst[currencies_stocks[0]]['Date'].values
for i in range(1,len(currencies_stocks)):
    time = np.intersect1d(time, currencies_inst[currencies_stocks[i]]['Date'].values)
    print(time.shape)


# In[55]:

for i in currencies_stocks:
    currencies_inst[i] = currencies_inst[i][currencies_inst[i]['Date'].isin(time)]
    
for i in currencies_stocks:
    currencies_inst[i].reset_index(inplace=True)


# In[56]:

for i in commodities_stocks1:
    print(commodities_inst[i]['Low'].isnull().sum())


# In[57]:

for i in currencies_stocks:
    if(currencies_inst[i]['Low'].isnull().sum()!=0):
        index = currencies_inst[i][currencies_inst[i]['Low'].isnull()==True].index

for i in currencies_stocks:
    currencies_inst[i] = currencies_inst[i].drop(currencies_inst[i].index[index])
    
for i in currencies_stocks:
    currencies_inst[i]['Open_inc'] = np.zeros(currencies_inst[i].shape[0])
    currencies_inst[i]['High_inc'] = np.zeros(currencies_inst[i].shape[0])
    currencies_inst[i]['Low_inc'] = np.zeros(currencies_inst[i].shape[0])
    currencies_inst[i]['Open_inc'][1:] = currencies_inst[i]['Open'][1:].values-currencies_inst[i]['Open'][0:-1].values
    currencies_inst[i]['High_inc'][1:] = currencies_inst[i]['High'][1:].values-currencies_inst[i]['High'][0:-1].values
    currencies_inst[i]['Low_inc'][1:] = (currencies_inst[i]['Low'][1:].values-currencies_inst[i]['Low'][0:-1].values)/(currencies_inst[i]['Low'][0:-1].values)
    
corr_comm = np.eye(len(currencies_stocks))
for i in range(len(currencies_stocks)-1):
    for j in range(i+1,len(currencies_stocks)):
        corr_comm[i][j] = currencies_inst[currencies_stocks[i]]['Low_inc'].corr(currencies_inst[currencies_stocks[j]]['Low_inc'], method='pearson')
        
corr_comm = np.eye(len(currencies_stocks))
for i in range(len(currencies_stocks)-1):
    for j in range(i+1,len(currencies_stocks)):
        corr_comm[i][j] = currencies_inst[currencies_stocks[i]]['Low_inc'].corr(currencies_inst[currencies_stocks[j]]['Low_inc'], method='pearson')


# In[58]:

edge_model = covariance.GraphLassoCV()
X = result.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

currencies_meta = np.asarray(currencies_meta)

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(currencies_meta[labels == i])))


# In[59]:

node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T


# In[60]:

plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(currencies_meta, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.savefig('Currencies.png')
plt.show()


# In[ ]:

currencies_stocks


# In[61]:

equities = pd.read_csv('Equities_data.csv')


# In[62]:

equitiesmeta = pd.read_csv('Equities_metadata.csv')


# In[63]:

equities_stocks = np.unique(equities['Ticker'])


# In[74]:

equities_meta = equitiesmeta['SECURITY_NAME'].values


# In[67]:

equities_inst = {}
for i in equities_stocks:
    equities_inst[i] = []


# In[68]:

equities_stocks[0]


# In[69]:

for i in equities_stocks: 
    equities_inst[i] = equities[equities['Ticker']==i]


# In[70]:

time = equities_inst[equities_stocks[0]]['Date'].values
for i in range(1,len(equities_stocks)):
    time = np.intersect1d(time,equities_inst[equities_stocks[i]]['Date'].values)
    print(time.shape)


# In[71]:

for i in equities_stocks:
    equities_inst[i] = equities_inst[i][equities_inst[i]['Date'].isin(time)]


# In[72]:

for i in equities_stocks:
    equities_inst[i].reset_index(inplace=True)


# In[73]:

for i in equities_stocks:
    print(equities_inst[i]['Low'].isnull().sum())


# In[ ]:

for i in equities_stocks:
    if(equities_inst[i]['Low'].isnull().sum()!=0):
        index = equities_inst[i][equities_inst[i]['Low'].isnull()==True].index

print(index)


# In[ ]:

for i in equities_stocks:
    equities_inst[i] = equities_inst[i].drop(equities_inst[i].index[index])


# In[75]:

for i in equities_stocks:
    #equities_inst[i]['Open_inc'] = np.zeros(equities_inst[i].shape[0])
    #equities_inst[i]['High_inc'] = np.zeros(equities_inst[i].shape[0])
    equities_inst[i]['Low_inc'] = np.zeros(equities_inst[i].shape[0])
    #equities_inst[i]['Open_inc'][1:] = equities_inst[i]['Open'][1:].values-equities_inst[i]['Open'][0:-1].values
    #equities_inst[i]['High_inc'][1:] = equities_inst[i]['High'][1:].values-equities_inst[i]['High'][0:-1].values
    equities_inst[i]['Low_inc'][1:] = (equities_inst[i]['Low'][1:].values-equities_inst[i]['Low'][0:-1].values)/(equities_inst[i]['Low'][0:-1].values)
    


# In[82]:

equities_stocks


# In[84]:

equities_inst['000027_CH_Equity']

corr_comm = np.eye(len(equities_stocks))
for i in range(len(equities_stocks)-1):
    for j in range(i+1,len(equities_stocks)):
        corr_comm[i][j] = equities_inst[equities_stocks[i]]['Low_inc'].corr(equities_inst[equities_stocks[j]]['Low_inc'], method='pearson')
        
# In[81]:

for i in range(len(equities_stocks)):
    col = equities_inst[equities_stocks[i]]['Low_inc'].values[1:]
    print(col)
    if(i == 0):
        result =  col
    else:
        result = np.vstack((result, col))


# In[76]:

edge_model = covariance.GraphLassoCV()
X = result.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

equities_meta = np.asarray(equities_meta)

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(equities_meta[labels == i])))


# In[77]:

node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T


# In[78]:

plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(equities_meta, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.savefig('Currencies.png')
plt.show()


# In[ ]:




# In[85]:

pm= pd.read_csv('metals.csv', parse_dates='Date')


# In[86]:

pmmeta = pd.read_csv('Precious_Metals_metadata.csv')


# In[87]:

pm_stocks = np.unique(pm['Ticker'])


# In[88]:

pm_meta = pmmeta['Long Description']


# In[ ]:

pmmeta


# In[89]:

pm_inst = {}
for i in pm_stocks:
    pm_inst[i] = []

for i in pm_stocks: 
    pm_inst[i] = pm[pm['Ticker']==i]


# In[90]:

time = pm_inst[pm_stocks[0]]['Date'].values
for i in range(1,len(pm_stocks)):
    time = np.intersect1d(time,pm_inst[pm_stocks[i]]['Date'].values)
    print(time.shape)


# In[91]:

for i in pm_stocks:
    pm_inst[i] = pm_inst[i][pm_inst[i]['Date'].isin(time)]
    


# In[92]:

for i in pm_stocks:
    pm_inst[i].reset_index(inplace=True)


# In[ ]:

pm


# In[ ]:

pmmeta


# In[93]:

for i in pm_stocks:
    print(pm_inst[i]['Quote'].isnull().sum())


# In[ ]:

pm

for i in pm_stocks:
    if(pm_inst[i]['Quote'].isnull().sum()!=0):
        index = pm_inst[i][pm_inst[i]['Quote'].isnull()==True].index
        print(len(index))

#for i in equities_stocks:
#    equities_inst[i] = equities_inst[i].drop(equities_inst[i][index])

# In[ ]:

pm_inst['XPT_BGN_Curncy']['Quote']


# In[94]:

for i in pm_stocks:
    pm_inst[i]['Quote_inc'] = np.zeros(pm_inst[i].shape[0])
    pm_inst[i]['Quote_inc'][1:] = (pm_inst[i]['Quote'][1:].values-pm_inst[i]['Quote'][0:-1].values)/(pm_inst[i]['Quote'][0:-1].values)

corr_comm = np.eye(len(pm_stocks))
for i in range(len(pm_stocks)-1):
    for j in range(i+1,len(pm_stocks)):
        corr_comm[i][j] = pm_inst[pm_stocks[i]]['Quote'].corr(pm_inst[pm_stocks[j]]['Quote'], method='pearson')
        
corr_comm = np.eye(len(pm_stocks))
for i in range(len(pm_stocks)-1):
    for j in range(i+1,len(pm_stocks)):
        corr_comm[i][j] = pm_inst[pm_stocks[i]]['Quote'].corr(pm_inst[pm_stocks[j]]['Quote'], method='pearson')
# In[ ]:

pm_inst['XPT_BGN_Curncy']['Quote']


# In[95]:

for i in range(len(pm_stocks)):
    col = pm_inst[pm_stocks[i]]['Quote_inc'][1:].values
    print(col)
    if(i == 0):
        result =  col
    else:
        result = np.vstack((result, col))


# In[102]:

for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        result[i][j] = result[i][j].astype(np.int)


# In[105]:

edge_model = covariance.GraphLassoCV()
X = result.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

pm_meta = np.asarray(pm_meta)

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(pm_meta[labels == i])))


# In[97]:

np.any(np.isnan(result.T))


# In[98]:

np.all(np.isfinite(result.T))


# In[ ]:

node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T


# In[ ]:

plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(equities_meta, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.savefig('Currencies.png')
plt.show()


# In[106]:

econ = pd.read_csv('Economic_Indicators_data.csv',sep=';')


# In[107]:

econmeta = pd.read_csv('Economic_Indicators_metadata.csv')


# In[108]:

econ_stocks = np.unique(econ['Ticker'])


# In[109]:

econ


# In[110]:

econ_inst = {}
for i in econ_stocks:
    econ_inst[i] = []

for i in econ_stocks: 
    econ_inst[i] = econ[econ['Ticker']==i]


# In[111]:

econ_inst


# In[112]:

time = econ_inst[econ_stocks[0]]['Date'].values
for i in range(1,len(pm_stocks)):
    time = np.intersect1d(time,econ_inst[econ_stocks[i]]['Date'].values)
    print(time.shape)


# In[ ]:



