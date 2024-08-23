import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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
#bins = np.arange(4000, 170000, 10000) # shows full data range well
bins = np.arange(3000, 15000, 1400)
data['flow_bin'] = pd.cut(data['flow (cfs)'], bins=bins)

# Convert intervals to strings for mapping
data['flow_bin_str'] = data['flow_bin'].astype(str)

# Create a dictionary with the actual interval labels
interval_labels = {str(interval): f"({int(interval.left)}, {int(interval.right)}]" for interval in data['flow_bin'].cat.categories}

# Drop rows with NaN values in 'flow_bin'
data = data.dropna(subset=['flow_bin'])

# Generate the ridgeline plot
fig, ax = plt.subplots(figsize=(10, 8))

# Get unique bins and sort them
unique_bins = sorted(data['flow_bin'].unique())

# Set the y_offset for each bin
y_offset = 1

# Define a color map based on the flow range
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_bins)))

# Plot density for each bin
for i, bin in enumerate(unique_bins):
    # Select data for the current bin
    bin_salinity = data[data['flow_bin'] == bin]['CB2.2'].dropna()

    # Calculate density
    density = gaussian_kde(bin_salinity)
    xs = np.linspace(bin_salinity.min(), bin_salinity.max(), 200)
    ys = density(xs)

    # Normalize ys for visualization purposes
    ys = ys / ys.max()

    # Plot the density curve
    ax.fill_between(xs, ys + i * y_offset, i * y_offset, color=colors[i], alpha=0.6)

    # Label the bin
 #   ax.text(xs.min(), i * y_offset + y_offset / 2, interval_labels[str(bin)], verticalalignment='center')

# Set labels and title
ax.set_xlabel('Salinity (psu)')
ax.set_ylabel('Flow (cfs)')
ax.set_yticks(np.arange(len(unique_bins)) * y_offset)
ax.set_yticklabels([interval_labels[str(bin)] for bin in unique_bins], va='center')
ax.set_title('Salinity Distribution Across Flow')

plt.show()
