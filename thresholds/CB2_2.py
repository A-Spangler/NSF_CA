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

# make flow and salinity dataframes
x = q['flow (cfs)']
y = cb2_2['CB2.2']

#drop rows where flow is over threshold
dict = {'x': x, 'y': y}
df = pd.DataFrame(dict)
df = df.drop(df[df['x'] >= 200000].index)

#rearrange back into x and y df's
x = df['x']
y = df['y']

# Fit cubic regression model
degree = 2  # Degree of the polynomial regression
coefficients = np.polyfit(x, y, degree)
poly = np.poly1d(coefficients)

# Make regression model
range_values = np.linspace(min(x), max(x), 1000)
model_values = np.polyval(coefficients, range_values)

# Calculate 'predicted' y values
y_pred = np.polyval(coefficients, x)

# Calculate residuals
residuals = y - y_pred

# Calculate R-squared
SSE = SSE = np.sum((y - y_pred)**2)  # Regression sum of squares
SST = np.sum((y - np.mean(y))**2)    # Total sum of squares
r_squared = 1- (SSE / SST)
rmse = mean_squared_error(y, y_pred, squared=False)

# Calculate F-statistic
k = degree  # Number of predictors
n = len(y)  # Number of observations
f_stat = (r_squared / k) / ((1 - r_squared) / (n - k - 1))

# Calculate p-value from F-statistic
p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)

# Print R-squared, RMSE, and p-value
print("R-squared:", r_squared)
print("RMSE:", rmse)
print("P-value:", p_value)

## Create Confidence Intervals
# Define bootstrap resampling function
def bootstrap_resample(x, y, n_boot=1000):
    n = len(x) # number of points in dataset
    boot_coefs = [] # empty list to store coefs
    for _ in range(n_boot):
        indices = np.random.choice(n, n, replace=True) # randomly n sample x's from range 0 to n with replacement
        x_boot, y_boot = x.iloc[indices], y.iloc[indices] # select the corresponding value from the sampled index
        boot_coefs.append(np.polyfit(x_boot, y_boot, degree)) # fit polynomial to bootstrap vals, append coefs
    return np.array(boot_coefs)

# Do the bootstrap resampling
boot_coefs = bootstrap_resample(x, y)

# Calculate confidence intervals from bootstrap samples. Percentiles match 95% CI
conf_int_lower = np.percentile(boot_coefs, 2.5, axis=0)
conf_int_upper = np.percentile(boot_coefs, 97.5, axis=0)

# Plot data and regression line
plt.scatter(x, y, color='lightblue', label='Model Data')
plt.plot(range_values, model_values, color='purple', label='Polynomial Regression')

# Plot confidence interval of the model
plt.fill_between(range_values, np.polyval(conf_int_lower, range_values), np.polyval(conf_int_upper, range_values),
                 color='purple', alpha=0.2, label='95% Confidence Interval')

# Plot drinking water limit
plt.axhline(y=7, color='c', label='Healthy Drinking Water Limit')  # Plot suggested limit

#plt.axvline(x=5925, color='goldenrod', label='Flow')  # Plot suggested limit

# flow = 14473 to meet 6psu goal

plt.xlabel('Flow (cfs)')
plt.ylabel('Salinity (psu)')
#plt.xlim(5920,5930)
#plt.ylim(6,8)
plt.title('Salinity Model - CB2.2')
plt.legend(loc='upper right')
#plt.show()
plt.savefig("CB2.svg")
