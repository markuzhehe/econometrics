#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:58:40 2024

@author: markyhehe
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("vote.csv")

# Define the dependent variable
Y = data['voteA']

# a. Regress voteA on democA, expendA, and expendB
X = data[['democA', 'expendA', 'expendB']]
X = sm.add_constant(X)  
model_a = sm.OLS(Y, X).fit()

# democA yields a coefficent of 4.391, with a p-value of 0.017 making the 
# statistic significant. this indicates that democrats are more likely to win. 

# b. Regress voteA on democA, expendA, expendB, and shareA
data['shareA'] = 100 * data['expendA'] / (data['expendA'] + data['expendB'])
X = data[['democA', 'expendA', 'expendB', 'shareA']]
X = sm.add_constant(X)  
model_b = sm.OLS(Y, X).fit()

# c. Regress voteA on democA, shareA
X = data[['democA', 'shareA']]
X = sm.add_constant(X)  
model_c = sm.OLS(Y, X).fit()

# d. Compare the models from (a)-(c) using adjusted R squared
print(f"Adjusted R-squared for model a: {model_a.rsquared_adj}")
print(f"Adjusted R-squared for model b: {model_b.rsquared_adj}")
print(f"Adjusted R-squared for model c: {model_c.rsquared_adj}")

# the adjusted R squareds for models A, B, and C were 0.5375104607250247, 
# 0.8569829283237198, 0.8544566280148624 respectively. 

# e. Generate a dummy variable for whether Candidate A won the election, and 
# call it winA
data['winA'] = np.where(data['voteA'] >= 50, 1, 0)
Y = data['winA']
X = data[['democA', 'expendA', 'expendB']]
X = sm.add_constant(X)
model_e = sm.Logit(Y, X).fit()

# holding expidentures constant, although democA is positive indicating a 
# higher chance of winning, it has a coefficent of 0.0998 meaning that holding
# expidentures constant it's chances increases 0.0998 per unit increase
# The minimum and maximum values from the regression are 0 and 1

# f. Repeat part e using heteroskedasticity robust standard errors
model_f = sm.Logit(Y, X).fit(cov_type='HC0')

# Print the summary of each model
print(model_a.summary())
print(model_b.summary())
print(model_c.summary())
print(model_e.summary())
print(model_f.summary())
