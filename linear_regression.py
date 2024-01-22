#!/bin/env python3

import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

df = pd.read_csv('homeprices.csv')
new_df = df.drop('price', axis='columns')
price = df.price

# Create linear regression object
linear_reg_model = linear_model.LinearRegression()

# save model before training
with open('model_b4_training','wb') as file:
    pickle.dump(linear_reg_model,file)

# train model
linear_reg_model.fit(new_df, price)

# save model after training
with open('model_after_training','wb') as file:
    pickle.dump(linear_reg_model,file)

linear_reg_model.predict([[3300]])

