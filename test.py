import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import os
import sklearn.metrics
import warnings
import utils

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Dropout, Embedding
from keras.losses import CategoricalCrossentropy
from keras.optimizers import SGD, Adam

print(os.listdir())

warnings.filterwarnings('ignore')

dataset_0102 = utils.preprocess('0102')
print(dataset_0102.describe())
dataset_0304 = utils.preprocess('0304')
print(dataset_0304.describe())

merged_dataset = pd.concat([dataset_0102,dataset_0304],axis=0)
merged_dataset = merged_dataset.reset_index(drop=True)
age_split = utils.spilt_by_age(merged_dataset)
low_age_dataset = age_split["low_age_dataset"]
mid_age_dataset = age_split["mid_age_dataset"]
high_age_dataset = age_split["high_age_dataset"]


# # analysing the target var
# KIQ400 = merged_dataset['KIQ400']
# print(merged_dataset.describe())
#
# # check correlation between columns
# print(merged_dataset.corr()['KIQ400'].abs().sort_values(ascending=False))
#
# value_counts = KIQ400.value_counts().sort_index()
#
# sns.barplot(x=value_counts.index, y=value_counts.values, orient='h')
#
# plt.xticks([0, 1, 2, 3, 4])
# plt.xlabel('KIQ400 level')
# plt.ylabel('Count')
# plt.title('KIQ400 level Count')
#
# plt.show()
#
# rcParams['figure.figsize'] = 20, 14
# plt.matshow(merged_dataset.corr())
# plt.yticks(np.arange(merged_dataset.shape[1]), merged_dataset.columns)
# plt.xticks(np.arange(merged_dataset.shape[1]), merged_dataset.columns)
# plt.colorbar()
#
# plt.show()

# # analysing the target var
# KIQ400 = merged_dataset['KIQ400']
# # merged_dataset['KIQ400'] = KIQ400
#
# print(merged_dataset.describe())
#
# # check correlation between columns
# print(merged_dataset.corr()['KIQ400'].abs().sort_values(ascending=False))
#
# value_counts = KIQ400.value_counts().sort_index()
#
# sns.barplot(x=value_counts.index, y=value_counts.values, orient='h')

# plt.xticks([0, 1, 2, 3, 4])
# plt.xlabel('KIQ400 level')
# plt.ylabel('Count')
# plt.title('KIQ400 level Count')
#
# # plt.show()
#
# rcParams['figure.figsize'] = 20, 14
# plt.matshow(merged_dataset.corr())
# plt.yticks(np.arange(merged_dataset.shape[1]), merged_dataset.columns)
# plt.xticks(np.arange(merged_dataset.shape[1]), merged_dataset.columns)
# plt.colorbar()

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

# predictors = merged_dataset.drop("KIQ400", axis=1)
# # normalization
# # for col in predictors.columns:
# #     predictors[col] = (predictors[col] - predictors[col].min()) / (predictors[col].max() - predictors[col].min())
#
# # print(predictors)
# target = merged_dataset["KIQ400"]
#
# X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=32)
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
#
#
# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=6))
# model.add(Dense(1, activation='sigmoid'))
# # model.add(Dense(10, activation='softmax'))
# # model.add(Dropout(0.5))
# # model.add(LSTM(128,activation='relu',input_shape=(2,1)))
# # model.add(Dense(1))
# # model.add(Embedding(4, output_dim=4))
# # model.add(LSTM(4))
# # model.add(Dropout(0.5))
# # model.add(Dense(1, activation='sigmoid'))
#
# model.summary()
#
# sgd = SGD(lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-6)
# adam = Adam(learning_rate=0.01, weight_decay=1e-6)
# model.compile(loss=CategoricalCrossentropy(), optimizer=adam, metrics=['accuracy'])
#
# model.fit(X_train, Y_train, epochs=300, batch_size=16)
# # score = model.evaluate(X_test,Y_test,batch_size=128)
#
# Y_pred_nn = model.predict(X_test)
#
# rounded = [round(x[0]) for x in Y_pred_nn]
#
# Y_pred_nn = rounded
#
# score_nn = round(sklearn.metrics.accuracy_score(Y_pred_nn, Y_test) * 100, 2)
#
# print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")

# Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.
