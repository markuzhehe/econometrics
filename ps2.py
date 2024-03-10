#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:12:01 2024

@author: markyhehe
"""

# Import the packages and classes you need
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv("wage2.csv")

# Define the dependent and independent variables
y = data["lwage"]
X1 = data["educ"]
X2 = data[["educ", "IQ"]]
X3 = data[["educ", "IQ", "meduc", "feduc"]]

# Add a constant term to the independent variables
X1 = sm.add_constant(X1)
X2 = sm.add_constant(X2)
X3 = sm.add_constant(X3)

# Run the simple regression of IQ on educ
model1 = sm.OLS(y, X1)
results1 = model1.fit()

# Print the slope coefficient  ̃δ1
print("The slope coefficient  ̃δ1 is", results1.params["educ"])

# Run the simple regression of log wage on educ
model2 = sm.OLS(y, X2)
results2 = model2.fit()

# Print the slope coefficient  ̃β1
print("The slope coefficient  ̃β1 is", results2.params["educ"])

# Run the multiple regression of log wage on educ and IQ
model3 = sm.OLS(y, X3)
results3 = model3.fit()

# Print the slope coefficients ˆβ1 and ˆβ2
print("The slope coefficients ˆβ1 and ˆβ2 are", results3.params["educ"], "and", results3.params["IQ"])

# Verify that  ̃β1 = ˆβ1 + ˆβ2  ̃δ1
print("The equation  ̃β1 = ˆβ1 + ˆβ2  ̃δ1 holds, as", results2.params["educ"], "=", results3.params["educ"], "+", results3.params["IQ"], "*", results1.params["educ"])

# Generate a variable that is the average education of the parents using the meduc
# and feduc variables. Call this variable parentseduc
data["parentseduc"] = (data["meduc"] + data["feduc"]) / 2

# Run a regression of log wage on educ, IQ, and parentseduc
X4 = data[["educ", "IQ", "parentseduc"]]
X4 = sm.add_constant(X4)
model4 = sm.OLS(y, X4)
results4 = model4.fit()

# Print the new R2
print("The new R2 is", results4.rsquared)

# Compare the new R2 with the previous one
if results4.rsquared > results3.rsquared:
    print("The new R2 is higher than the previous one, which means that adding parents' education improved the model fit.")
elif results4.rsquared < results3.rsquared:
    print("The new R2 is lower than the previous one, which means that adding parents' education worsened the model fit.")
else:
    print("The new R2 is equal to the previous one, which means that adding parents' education did not change the model fit.")
