import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# Import the excel sheets
cb2_2 = pd.read_excel(r'/Users/aas6791/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Research/NSF_CA/SalinityData/CB2_2.XLSX')
q = pd.read_excel(r'/Users/aas6791/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Research/NSF_CA/SalinityData/Susquehanna_RiverDischarge.XLSX')

# Adjust m3/s flow to cfs
q['flow (cfs)'] = q['flow (m3/s)'] * ((1/2.54)**3) * ((1/12)**3) * (100**3)

# Normalize the hourly salinity data into daily averages
cb2_2 = cb2_2.groupby(by=pd.Grouper(freq='D', key='time')).mean()
cb2_2 = cb2_2.reset_index()

# Make flow and salinity dataframes
x = q['flow (cfs)']
y = cb2_2['CB2.2']

# Define threshold
threshold = 100000

# Drop rows where flow is over threshold
df = pd.DataFrame({'x': x, 'y': y})
df_full = df.copy()
df_partial = df[df['x'] < threshold]

# Fit models for full and partial ranges
def fit_and_evaluate(df, degree):
    x = df['x']
    y = df['y']
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)
    y_pred = poly(x)
    r_squared = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    return poly, r_squared, rmse

degree = 2
poly_full, r2_full, rmse_full = fit_and_evaluate(df_full, degree)
poly_partial, r2_partial, rmse_partial = fit_and_evaluate(df_partial, degree)

# Define bootstrap resampling function
def bootstrap_resample(x, y, n_boot=1000):
    n = len(x)  # number of points in dataset
    boot_coefs = []  # empty list to store coefs
    for _ in range(n_boot):
        indices = np.random.choice(n, n, replace=True)  # randomly n sample x's from range 0 to n with replacement
        x_boot, y_boot = x.iloc[indices], y.iloc[indices]  # select the corresponding value from the sampled index
        boot_coefs.append(np.polyfit(x_boot, y_boot, degree))  # fit polynomial to bootstrap vals, append coefs
    return np.array(boot_coefs)

# Print R-squared and RMSE
print(f"Full Range R-squared: {r2_full:.4f}, RMSE: {rmse_full:.4f}")
print(f"Partial Range R-squared: {r2_partial:.4f}, RMSE: {rmse_partial:.4f}")

# Do the bootstrap resampling for partial range
boot_coefs_partial = bootstrap_resample(df_partial['x'], df_partial['y'])
boot_coefs_full = bootstrap_resample(df_full['x'], df_full['y'])

# Calculate confidence intervals from bootstrap samples. Percentiles match 95% CI
conf_int_lower_partial = np.percentile(boot_coefs_partial, 2.5, axis=0)
conf_int_upper_partial = np.percentile(boot_coefs_partial, 97.5, axis=0)
conf_int_lower_full = np.percentile(boot_coefs_full, 2.5, axis=0)
conf_int_upper_full = np.percentile(boot_coefs_full, 97.5, axis=0)



# Plot data and regression lines
plt.figure(figsize=(12, 6))
plt.scatter(df_full['x'], df_full['y'], color='lightblue', label='Data')
plt.plot(np.linspace(min(df_partial['x']), max(df_partial['x']), 1000), poly_partial(np.linspace(min(df_partial['x']), max(df_partial['x']), 1000)), color='purple', label=f'Partial Range Fit (R² = {r2_partial:.2f})')
plt.plot(np.linspace(min(df_full['x']), max(df_full['x']), 1000), poly_full(np.linspace(min(df_full['x']), max(df_full['x']), 1000)), color='orange', label=f'Full Range Fit (R² = {r2_full:.2f})')

# Plot confidence interval for partial range
#plt.fill_between(np.linspace(min(df_partial['x']), max(df_partial['x']), 1000), np.polyval(conf_int_lower_partial, np.linspace(min(df_partial['x']), max(df_partial['x']), 1000)), np.polyval(conf_int_upper_partial, np.linspace(min(df_partial['x']), max(df_partial['x']), 1000)), color='purple', alpha=0.1)
#plt.fill_between(np.linspace(min(df_full['x']), max(df_full['x']), 1000), np.polyval(conf_int_lower_full, np.linspace(min(df_full['x']), max(df_full['x']), 1000)), np.polyval(conf_int_upper_full, np.linspace(min(df_full['x']), max(df_full['x']), 1000)), color='orange', alpha=0.1)

plt.xlabel('Flow (cfs)')
plt.ylabel('Salinity (psu)')
plt.legend(loc='upper right')
plt.title('Salinity Model - CB2.2')
plt.show()