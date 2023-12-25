import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sklearn.metrics
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Dense,LSTM
from keras.losses import CategoricalCrossentropy

print(os.listdir())

warnings.filterwarnings('ignore')

dataset_ed = pd.read_csv('ED.csv')
dataset_b12 = pd.read_csv('B12.csv')

# merge datasets and remove SQEN col, reset index, drop unused cols
merged_dataset = pd.merge(dataset_ed, dataset_b12, on='SEQN', how='left')
print(merged_dataset)
merged_dataset = merged_dataset.loc[:, ['KIQ400', 'LBDFOLSI', 'LBDB12SI']]

# delete rows including na
merged_dataset = merged_dataset.dropna()
print(merged_dataset)
# based docs, drop ALQ values:77 99 777 999 ,'ALQ121' 'ALQ130','ALQ142',
#                                                 'ALQ270','ALQ280','ALQ290',
#                                                 'ALQ151','ALQ170'

for col in ['KIQ400']:
    drop_index = merged_dataset[merged_dataset[col].isin([7, 9])].index
    print(drop_index, 'deleted.')
    merged_dataset.drop(drop_index, inplace=True)

merged_dataset = merged_dataset.reset_index(drop=True)

print(merged_dataset)

# LBDFOLSI = merged_dataset['LBDFOLSI']
# merged_dataset['LBDFOLSI'] = LBDFOLSI.map(lambda x: 0 if x < 6.8 else 1)
# LBDB12SI = merged_dataset['LBDB12SI']
# merged_dataset['LBDB12SI'] = LBDB12SI.map(lambda x: 0 if x < 156 else 1)

print(merged_dataset)

# analysing the target var
KIQ400 = merged_dataset['KIQ400']
# merged_dataset['KIQ400'] = KIQ400

# check correlation between columns
print(merged_dataset.corr()['KIQ400'].abs().sort_values(ascending=False))

value_counts = KIQ400.value_counts().sort_index()

sns.barplot(x=value_counts.index, y=value_counts.values, orient='h')

plt.xticks([0, 1, 2, 3, 4])
plt.xlabel('KIQ400 level')
plt.ylabel('Count')
plt.title('KIQ400 level Count')

plt.show()

# get counts
# LBDB12SI = merged_dataset['LBDB12SI']
# value_counts = LBDB12SI.value_counts().sort_index()
#
# sns.barplot(x=value_counts.index, y=value_counts.values, orient='h')
#
# plt.xticks([0, 1])
# plt.xlabel('LBDB12SI level')
# plt.ylabel('Count')
# plt.title('LBDB12SI level Count')
# plt.show()
#
# LBDFOLSI = merged_dataset['LBDFOLSI']
# value_counts = LBDFOLSI.value_counts().sort_index()
#
# sns.barplot(x=value_counts.index, y=value_counts.values, orient='h')
#
# plt.xticks([0, 1])
# plt.xlabel('LBDFOLSI level')
# plt.ylabel('Count')
# plt.title('LBDFOLSI level Count')
# plt.show()

predictors = merged_dataset.drop("KIQ400", axis=1)
# normalization
# predictors = (predictors - predictors.min()) / (predictors.max()-predictors.min())
# print(predictors)
target = merged_dataset["KIQ400"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model = Sequential()
model.add(LSTM(128,activation='relu',input_shape=(2,1)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(X_train, Y_train, epochs=300)

Y_pred_nn = model.predict(X_test)

rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded

score_nn = round(sklearn.metrics.accuracy_score(Y_pred_nn, Y_test) * 100, 2)

print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

# Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.
