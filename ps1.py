import pandas as pd
auto_data = pd.read_csv("auto.csv")
auto_data.describe()
auto_data[["weight", "length", "price", "mpg", "foreign"]].describe()
auto_data["gpm"]=1/auto_data["mpg"]
auto_data[["mpg","gpm"]]
auto_data[["mpg","gpm"]].describe()

print("Calculated by hand", 1/21.297297)
print("Calculated in python", 1/auto_data["mpg"].mean())

import matplotlib.pyplot as plt
# Creating a scatter plot
3
plt.scatter(auto_data["weight"], auto_data["mpg"])
# Adding title and labels
plt.title("Scatter Plot of Weight vs MPG")
plt.xlabel("Weight")
plt.ylabel("MPG")
# Showing the plot
plt.show()
auto_data[["mpg","weight"]].corr()
foreign_cars=auto_data[auto_data["foreign"]=="Foreign"]
domestic_cars=auto_data[auto_data["foreign"]=="Domestic"]

print("Average price of foreign cars is:",foreign_cars["price"].mean())
print("Average price of domestic cars is:",domestic_cars["price"].mean())

import statsmodels.api as sm
X = sm.add_constant(auto_data["mpg"]) # Independent variable (with constant)
y = auto_data["price"] # Dependent variable
model1 = sm.OLS(y, X).fit()

print(model1.summary())

X_gpm = sm.add_constant(auto_data["gpm"])  # Independent variable (with constant)
y_gpm = auto_data["price"]  # Dependent variable
model_gpm = sm.OLS(y_gpm, X_gpm).fit()

# Display the summary for gpm
print(model_gpm.summary())

gpm_stats = auto_data["gpm"].describe()
mpg_stats = auto_data["mpg"].describe()

# Display the statistics
print("Sample Average:", gpm_stats["mean"])
print("Standard Deviation:", gpm_stats["std"])
print("Minimum:", gpm_stats["min"])
print("Maximum:", gpm_stats["max"])

print("Sample Average mpg:", mpg_stats["mean"])

mean_mpg = auto_data["mpg"].mean()
mean_weight = auto_data["weight"].mean()

# Calculate covariance
covariance = ((auto_data["mpg"] - mean_mpg) * (auto_data["weight"] - mean_weight)).sum()

# Calculate standard deviations
std_mpg = auto_data["mpg"].std()
std_weight = auto_data["weight"].std()

# Calculate correlation coefficient
correlation_coefficient = covariance / (std_mpg * std_weight)

print("Correlation Coefficient between mpg and weight:", correlation_coefficient)