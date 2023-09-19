# Problem statement

A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia.
 
The company is looking at prospective properties to buy to enter the market. You are required to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not.

The company wants to know:
- `Which variables are significant in predicting the price of a house?`, and
- `How well those variables describe the price of a house?`.
- Also, `determine the optimal value of lambda for ridge and lasso regression.`

# Business Goals
You are required to model the price of houses with the available independent variables. This model will then be used by the management to understand how exactly the prices vary with the variables. <br>
They can accordingly manipulate the strategy of the firm and concentrate on areas that will yield high returns. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

# Libraries used
- Pandas - `1.5.3`
- Numpy - `1.24.3`
- Matplotlib - `3.7.1`
- Seaborn - `0.12.2`
- Sklearn - `1.2.2`

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn import linear_model, metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import scale, StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
```

# Contact
Author: Sudarshan | `sudarshan_g@outlook.com`

