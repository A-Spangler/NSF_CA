import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

# Import the excel sheets
hdg = pd.read_excel(r'/Users/aas6791/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Research/NSF_CA/SalinityData/Havre_de_Grace.XLSX')
q = pd.read_excel(r'/Users/aas6791/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Research/NSF_CA/SalinityData/Susquehanna_RiverDischarge.XLSX')

# Adjust m3/s flow to cfs -> 1 in/ 2.54 cm, 1 ft/12 in, 100cm/m
q['flow (cfs)'] = q['flow (m3/s)'] * ((1/2.54)**3) * ((1/12)**3) * (100**3)

# Normalize the hourly salinity data into daily averages
hdg = hdg.groupby(by=pd.Grouper(freq='D', key='time')).mean()
hdg = hdg.reset_index()

x = q['flow (cfs)']
y = hdg['Drinking water intake at Havre de Grace']

#drop rows where flow is over 100000
dict = {'x': x, 'y': y}
df = pd.DataFrame(dict)
df = df.drop(df[df['x'] >= 50000].index)

#rearrange into x and y df's
x = df['x']
y = df['y']

# Fit cubic regression model
degree = 2  # Degree of the polynomial regression
coefficients = np.polyfit(x, y, degree)
poly = np.poly1d(coefficients)

# Make regression model
range_values = np.linspace(min(x), max(x), 1000)
model_values = np.polyval(coefficients, range_values)
# Calculate predicted y values
y_pred = np.polyval(coefficients, x)

# Calculate residuals
residuals = y - y_pred

# Calculate R-squared
SSR = np.sum((y_pred - np.mean(y))**2)  # Regression sum of squares
SST = np.sum((y - np.mean(y))**2)        # Total sum of squares
r_squared = SSR / SST

# Print R-squared
print("R-squared:", r_squared)

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
plt.axhline(y= 1 , color='c', label='Healthy Drinking Water Limit')  # Plot suggested limit

plt.xlabel('Flow (cfs)')
plt.ylabel('Salinity (psu)')
#plt.ylim(-2,12)
plt.title('Salinity Model - HDG')
plt.legend(loc='upper right')
plt.show()
#plt.savefig("HDG_NotValid.svg")