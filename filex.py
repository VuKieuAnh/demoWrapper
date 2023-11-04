import pandas as pd
from sklearn import metrics
import numpy as np

from sklearn.model_selection import train_test_split # Import train_test_split function
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression



file_name="adenocarcinoma.csv"
file_path="data/"+file_name
data = df = pd.read_csv(file_path)
df.columns.values[0] = "class"
x = df.iloc[:, df.columns != 'class']
y = df[['class']]
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

# Removing highly correlated features (here > .9)
correlated_features = set()
correlation_matrix = df.corr()

threshold = 0.90

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

# Exclusion of identified features
trainX_clean = trainX.drop(labels=correlated_features, axis=1)
testX_clean = testX.drop(labels=correlated_features, axis=1)

trainX_clean.shape, testX_clean.shape
from mlxtend.feature_selection import SequentialFeatureSelector
selector = SelectKBest(score_func=f_regression, k=20)

selector.fit(trainX_clean, trainY)

vector_names = list(trainX_clean.columns[selector.get_support(indices=True)])
print("vector_names",vector_names)

trainX_best = trainX_clean[vector_names]
testX_best = testX_clean[vector_names]

# print(trainX_best.shape)
# print(testX_best.shape)

trainX_reduced = trainX_best.iloc[0:10000,]
testX_reduced = testX_best.iloc[0:10000,]
trainY_reduced = trainY.iloc[0:10000,]
testY_reduced = testY.iloc[0:10000,]

# print(trainX_reduced.shape)
# print(testX_reduced.shape)
# print(trainY_reduced.shape)
# print(testY_reduced.shape)

feature_selector = SequentialFeatureSelector(RandomForestRegressor(n_jobs=-1),
           k_features=5,
           forward=True,
           verbose=2,
           scoring='r2',
           cv=4)

features = feature_selector.fit(np.array(trainX_reduced), trainY_reduced)
filtered_features= trainX_reduced.columns[list(features.k_feature_idx_)]
# print(filtered_features)

New_train_x = trainX_reduced[filtered_features]
New_test_x = testX_reduced[filtered_features]

feature_selector = SequentialFeatureSelector(RandomForestRegressor(n_jobs=-1),
           k_features=99,
           forward=False,
           verbose=2,
           scoring='r2',
           cv=4)
features = feature_selector.fit(np.array(trainX_reduced), trainY_reduced)
filtered_features= trainX_reduced.columns[list(features.k_feature_idx_)]
print("tinh nang dc chon", filtered_features)

