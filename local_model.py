import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib

#Load data
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.target 

#Split Model
X = df.drop(['MEDV'], axis = 1) 
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 42)

#Model Creation
lm = LinearRegression()
lm.fit(X_train,y_train)


with open('model.joblib', 'wb') as f:
    joblib.dump(lm,f)


with open('model.joblib', 'rb') as f:
    predictor = joblib.load(f)

print("Testing following input: ")
print(X_test[0:1])
sampInput = [[0.09178, 0.0, 4.05, 0.0, 0.51, 6.416, 84.1, 2.6463, 5.0, 296.0, 16.6, 395.5, 9.04]]
print(type(sampInput))
print(predictor.predict(sampInput))