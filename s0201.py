
# coding: utf-8

# # Seeing the Data Science Pipeline in Action Using Python


## Loading the Data
# Boston data set contains housing prices and other facts about houses in the Boston area


from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data,boston.target


# ## Training a Model
# The problem is to predict the price of the house according to other parameters, then, it is a Regression Problem

from sklearn.linear_model import LinearRegression
hypothesis = LinearRegression(normalize=True)
hypothesis.fit(X,y)


# ## Viewing a Result
# There is no much to view in this case only the coefficient output from the linear regression analysis

print hypothesis.coef_




