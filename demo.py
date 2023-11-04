import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

#for chapter 4.2.1
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
santandar_data = pd.read_csv("data/santander.csv")
# print(santandar_data.shape)


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    # print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
    #                                                           "There are " + str(mis_val_table_ren_columns.shape[0]) +
    #       " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
missing_values_table(santandar_data)
# Only numerical variables are considered here
num_col = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(santandar_data.select_dtypes(include=num_col).columns)
santandar = santandar_data[numerical_columns]

# Train / Test Split
x = santandar.drop(['ID', 'TARGET'], axis=1)
y = santandar['TARGET']
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)

# Removing highly correlated features (here > .9)
correlated_features = set()
correlation_matrix = santandar.corr()

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
           k_features=6,
           forward=False,
           verbose=2,
           scoring='r2',
           cv=4)

features = feature_selector.fit(np.array(trainX_reduced), trainY_reduced)
filtered_features= trainX_reduced.columns[list(features.k_feature_idx_)]
print("tinh nang dc chon",filtered_features)
New_train_x = trainX_reduced[filtered_features]
New_test_x = testX_reduced[filtered_features]


# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
#
# rfe = RFE(lr, n_features_to_select=5)
# rfe.fit(trainX_reduced,trainY_reduced)
#
# rfe.support_
#
# rfe.ranking_
#
# Columns = trainX_reduced.columns
# RFE_support = rfe.support_
# RFE_ranking = rfe.ranking_
#
# dataset = pd.DataFrame({'Columns': Columns, 'RFE_support': RFE_support, 'RFE_ranking': RFE_ranking}, columns=['Columns', 'RFE_support', 'RFE_ranking'])
# dataset
# df = dataset[(dataset["RFE_support"] == True) & (dataset["RFE_ranking"] == 1)]
# filtered_features = df['Columns']
# # print("filtered_features", filtered_features)
# New_train_x = trainX_reduced[filtered_features]
# New_test_x = testX_reduced[filtered_features]