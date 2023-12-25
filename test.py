import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print(os.listdir())

import warnings

warnings.filterwarnings('ignore')

dataset_alq = pd.read_csv('ALQ.csv')
dataset_trigly = pd.read_csv('TRIGLY.csv')

# delete rows including na
dataset_alq = dataset_alq.dropna()
dataset_trigly = dataset_trigly.dropna()

# merge datasets and remove SQEN col, reset index, drop unused cols
merged_dataset = pd.merge(dataset_trigly, dataset_alq, on='SEQN', how='left').dropna()
merged_dataset = merged_dataset.drop(columns=['ALQ111','ALQ151', 'SEQN', 'LBDTRSI', 'LBDLDLSI',
                                              'LBDLDL', 'LBDLDLM', 'LBDLDMSI', 'LBDLDLN',
                                              'LBDLDNSI', 'WTSAFPRP'])

# based docs, drop ALQ values:77 99 777 999 ,'ALQ121' 'ALQ130','ALQ142',
#                                                 'ALQ270','ALQ280','ALQ290',
#                                                 'ALQ151','ALQ170'

for col in ['ALQ121', 'ALQ130', 'ALQ142', 'ALQ270', 'ALQ280', 'ALQ290', 'ALQ170']:
    drop_index = merged_dataset[merged_dataset[col].isin([77, 99, 777, 999])].index
    print(drop_index, 'deleted.')
    merged_dataset.drop(drop_index, inplace=True)

merged_dataset = merged_dataset.reset_index(drop=True)

print(merged_dataset)

# analysing the target var
LBXTR = merged_dataset['LBXTR']

# check correlation between columns
print(merged_dataset.corr()['LBXTR'].abs().sort_values(ascending=False))


def category_lbxtr(x):
    if x < 100:
        return 1
    elif x < 150:
        return 2
    else:
        return 3


LBXTR_preprocessed = LBXTR.map(category_lbxtr)
merged_dataset['LBXTR'] = LBXTR_preprocessed
value_counts = LBXTR_preprocessed.value_counts().sort_index()

sns.barplot(x=value_counts.index, y=value_counts.values, orient='h')

plt.xticks([0, 1, 2, 3])
plt.xlabel('LBXTR level')
plt.ylabel('Count')
plt.title('LBXTR level Count')

# plt.show()

predictors = merged_dataset.drop("LBXTR", axis=1)
target = merged_dataset["LBXTR"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(12, activation='relu', input_dim=7))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=300)

Y_pred_nn = model.predict(X_test)

rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded

score_nn = round(sklearn.metrics.accuracy_score(Y_pred_nn, Y_test) * 100, 2)

print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

# Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.
