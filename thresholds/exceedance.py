import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.metrics import mean_squared_error
from scipy import stats

# Import the excel sheets
cb2_2 = pd.read_excel(r'/Users/aas6791/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Research/NSF_CA/SalinityData/CB2_2.XLSX')
q = pd.read_excel(r'/Users/aas6791/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Research/NSF_CA/SalinityData/Susquehanna_RiverDischarge.XLSX')

# Adjust m3/s flow to cfs -> 1 in/ 2.54 cm, 1 ft/12 in, 100cm/m
q['flow (cfs)'] = q['flow (m3/s)'] * ((1/2.54)**3) * ((1/12)**3) * (100**3)

# Normalize the hourly salinity data into daily averages
cb2_2 = cb2_2.groupby(by=pd.Grouper(freq='D', key='time')).mean()
cb2_2 = cb2_2.reset_index()

# make flow and salinity dataframes, combine
x = q['flow (cfs)']
y = cb2_2['CB2.2']
df = pd.concat([x, y], axis = 1)

# sort the flow values greatest to least
df_sort = df.sort_values(by='flow (cfs)', ascending=False)

# assign each value a rank by resetting index, largest flow = rank 1
df_sort['rank'] = df_sort['flow (cfs)'].rank(ascending=False)
df_sort = df_sort.reset_index()
#print(df_sort)

# Define the number of events (n)
n = len(df_sort)

# Calculate exceedance probability (p) for each flow value
df_sort['p'] = 100 * (df_sort['rank'] / (n + 1))

#drop salinity values below threshold
df_drop = df_sort.drop(df_sort[df_sort['CB2.2'] <= 6].index)

# Plot, color by salinity range
fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axis to plot on

# Define color map
cmap = plt.cm.YlOrBr

# Scatter plot with colors based on CB2.2 (salinity) values. change to df_drop for full range
sc = ax.scatter(df_sort['p'], df_sort['flow (cfs)'], c=df_sort['CB2.2'], cmap=cmap)
ax.set_xlabel('Exceedance Probability (%)')
ax.set_ylabel('Flow (cfs)')
ax.set_title('Flow Duration')

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Salinity (psu)')

plt.show()
#plt.savefig('exceedance.svg')
