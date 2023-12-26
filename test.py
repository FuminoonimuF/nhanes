import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import os
import sklearn.metrics
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout, Embedding
from keras.losses import CategoricalCrossentropy
from keras.optimizers import SGD, Adam

print(os.listdir())

warnings.filterwarnings('ignore')

dataset_ed = pd.read_csv('ED.csv')
dataset_b12 = pd.read_csv('B12.csv')
dateset_bmx = pd.read_csv('BMX_B.csv')
dataset_demo = pd.read_csv('DEMO_B.csv')

# create interviewee info data
info_dataset = pd.merge(dataset_demo, dateset_bmx, on='SEQN', how='left')
info_dataset = info_dataset.loc[:, ['BMXWT', 'SEQN', 'BMXHT', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH1']]

# merge datasets and remove SQEN col, reset index, drop unused cols
merged_dataset = pd.merge(dataset_ed, dataset_b12, on='SEQN', how='left')
merged_dataset = merged_dataset.loc[:, ['KIQ400', 'LBDFOLSI', 'LBDB12SI', 'SEQN']]
merged_dataset = pd.merge(info_dataset, merged_dataset, on='SEQN', how='left')
merged_dataset = merged_dataset.drop(columns=['SEQN'])

print(merged_dataset)

# delete rows including na
merged_dataset = merged_dataset.dropna()
print(merged_dataset)
# based docs, drop ALQ values:77 99 777 999 ,'ALQ121' 'ALQ130','ALQ142',
#                                                 'ALQ270','ALQ280','ALQ290',
#                                                 'ALQ151','ALQ170'

# delete incorrect data
drop_index = merged_dataset[merged_dataset['KIQ400'].isin([7, 9])].index
print(drop_index, 'deleted.')
merged_dataset.drop(drop_index, inplace=True)

drop_index = merged_dataset[merged_dataset['LBDB12SI'] > 5000].index
print(drop_index, 'deleted')
merged_dataset.drop(drop_index, inplace=True)

drop_index = merged_dataset[merged_dataset['LBDFOLSI'] > 500].index
print(drop_index, 'deleted')
merged_dataset.drop(drop_index, inplace=True)

drop_index = merged_dataset[merged_dataset['RIAGENDR'] == 2].index
print(drop_index, 'deleted')
merged_dataset.drop(drop_index, inplace=True)
merged_dataset = merged_dataset.drop(columns=['RIAGENDR'])

merged_dataset = merged_dataset.reset_index(drop=True)

print(merged_dataset)

# LBDFOLSI = merged_dataset['LBDFOLSI']
# merged_dataset['LBDFOLSI'] = LBDFOLSI.map(lambda x: 0 if x < 6.8 else 1)
# LBDB12SI = merged_dataset['LBDB12SI']
# merged_dataset['LBDB12SI'] = LBDB12SI.map(lambda x: 0 if x < 156 else 1)

# analysing the target var
KIQ400 = merged_dataset['KIQ400']
# merged_dataset['KIQ400'] = KIQ400

print(merged_dataset.describe())

# check correlation between columns
print(merged_dataset.corr()['KIQ400'].abs().sort_values(ascending=False))

value_counts = KIQ400.value_counts().sort_index()

sns.barplot(x=value_counts.index, y=value_counts.values, orient='h')

plt.xticks([0, 1, 2, 3, 4])
plt.xlabel('KIQ400 level')
plt.ylabel('Count')
plt.title('KIQ400 level Count')

# plt.show()

rcParams['figure.figsize'] = 20, 14
plt.matshow(merged_dataset.corr())
plt.yticks(np.arange(merged_dataset.shape[1]), merged_dataset.columns)
plt.xticks(np.arange(merged_dataset.shape[1]), merged_dataset.columns)
plt.colorbar()

# plt.show()
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
# for col in predictors.columns:
#     predictors[col] = (predictors[col] - predictors[col].min()) / (predictors[col].max() - predictors[col].min())

# print(predictors)
target = merged_dataset["KIQ400"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


model = Sequential()
model.add(Dense(32, activation='relu', input_dim=6))
model.add(Dense(1, activation='sigmoid'))
# model.add(Dense(10, activation='softmax'))
# model.add(Dropout(0.5))
# model.add(LSTM(128,activation='relu',input_shape=(2,1)))
# model.add(Dense(1))
# model.add(Embedding(4, output_dim=4))
# model.add(LSTM(4))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

model.summary()

sgd = SGD(lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-6)
adam = Adam(learning_rate=0.01, weight_decay=1e-6)
model.compile(loss=CategoricalCrossentropy(), optimizer=adam, metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=300, batch_size=16)
# score = model.evaluate(X_test,Y_test,batch_size=128)

Y_pred_nn = model.predict(X_test)

rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded

score_nn = round(sklearn.metrics.accuracy_score(Y_pred_nn, Y_test) * 100, 2)

print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

# Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.
