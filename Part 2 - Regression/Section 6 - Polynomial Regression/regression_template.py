# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""




#predict a new salary of previous employee using linear regression
y_pred = regressor.predict(6.5)

#Visualizing the linear regression result
plt.scatter(X, y, color = 'red')
plt.title('Truth or bluff (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid )), color = 'blue')
plt.show() 



#Visualizing the linear regression result for higher resolution and sommther curve.
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, y, color = 'red')
plt.title('Truth or bluff (Regression model)')
plt.xlabel('Position level')
plt.ylabel('Salary')

plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid )), color = 'blue')
plt.show() 

