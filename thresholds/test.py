import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the excel sheets
cb2_2 = pd.read_excel(r'/Users/aas6791/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Research/NSF_CA/SalinityData/CB2_2.XLSX')
q = pd.read_excel(r'/Users/aas6791/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Research/NSF_CA/SalinityData/Susquehanna_RiverDischarge.XLSX')

# Adjust m³/s flow to cfs
q['flow (cfs)'] = q['flow (m3/s)'] * ((1 / 2.54) ** 3) * ((1 / 12) ** 3) * (100 ** 3)

# Normalize the hourly salinity data into daily averages
cb2_2 = cb2_2.groupby(by=pd.Grouper(freq='D', key='time')).mean()
cb2_2 = cb2_2.reset_index()

# Make flow and salinity dataframes, combine
x = q['flow (cfs)']
y = cb2_2['CB2.2']
data = pd.concat([x, y], axis=1)

# Define bins
bins = np.arange(4000, 170000, 10000)
data['flow_bin'] = pd.cut(data['flow (cfs)'], bins=bins)

# Convert intervals to strings for mapping
data['flow_bin_str'] = data['flow_bin'].astype(str)

# Create a dictionary with the actual interval labels
interval_labels = {str(interval): f"({int(interval.left)}, {int(interval.right)}]" for interval in data['flow_bin'].cat.categories}

# Drop rows with NaN values in 'flow_bin'
data = data.dropna(subset=['flow_bin'])

# Generate the ridgeline plot
# Sort data based on flow_bin
data = data.sort_values(by='flow_bin')

# Plot using seaborn
g = sns.FacetGrid(data, row='flow_bin_str', hue='flow_bin_str', aspect=4, height=4, palette='viridis')
g.map(sns.kdeplot, 'CB2.2', fill=True, alpha=0.6, linewidth=1.5)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define the y-axis labels
for ax, label in zip(g.axes.flat, data['flow_bin_str'].unique()):
    ax.text(-0.05, 0.2, interval_labels[label], fontweight='bold', color='black', ha='right', va='center', transform=ax.transAxes)

# Remove the original y-axis labels
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)

# Adjust the y-axis scale
for ax in g.axes.flat:
    ax.set_ylim(-0.05, 0.5)  # Adjust this as necessary

# Set labels and title
ax.set_xlim(0,12)
g.set_axis_labels('Salinity (psu)', '')
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Salinity Distribution Across Flow')

plt.show()
