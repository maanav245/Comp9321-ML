from cProfile import label
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot
from pyrsistent import inc
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle
from scipy.stats import pearsonr

from sklearn.datasets import make_regression
import seaborn as sns


def prepareData(df, df_test):
    df = df.drop(['WEEKDAY_APPR_PROCESS_START'], axis=1)
    df_test = df_test.drop(['WEEKDAY_APPR_PROCESS_START'], axis=1)

    col_list = df.columns
    adj_cols = []
    for col in col_list:
        if ('NAME' in col) or ('FLAG' in col) or ('TYPE' in col) or ('CODE_GENDER' in col) or ('FONDKAPREMONT_MODE' in col) or ('WALLSMATERIAL_MODE' in col) or ('EMERGENCYSTATE_MODE' in col):

            adj_cols.append(col)
    df = pd.get_dummies(df, columns=adj_cols)
    df_test = pd.get_dummies(df_test, columns=adj_cols)

    list1 = df.columns
    list2 = df_test.columns
    new_list = list(set(list1) ^ set(list2))
    print(new_list)

    for val in new_list:
        if val in list1:
            df = df.drop([val], axis=1)
        if val in list2:
            df_test = df_test.drop([val], axis=1)

    for column in df.columns:
        imputer = SimpleImputer(missing_values=np.nan,
                                strategy="mean")
        df[column] = imputer.fit_transform(df[[column]])
    for column in df_test.columns:
        imputer = SimpleImputer(missing_values=np.nan,
                                strategy="mean")
        df_test[column] = imputer.fit_transform(df_test[[column]])

    return df, df_test


def create_sets(df, algo, df_test, k_val):

    if algo == 0:
        income_x = df.drop(
            ['AMT_INCOME_TOTAL'], axis=1).values
        income_y = df['AMT_INCOME_TOTAL'].values
        income_x_t = df_test.drop(
            ['AMT_INCOME_TOTAL'], axis=1).values
        income_y_t = df_test['AMT_INCOME_TOTAL'].values
    else:
        income_x = df.drop(
            ['TARGET'], axis=1).values
        income_y = df['TARGET'].values
        income_x_t = df_test.drop(
            ['TARGET'], axis=1).values
        income_y_t = df_test['TARGET'].values

    fs = SelectKBest(score_func=f_regression, k=k_val)
    # learn relationship from training data
    fs.fit(income_x, income_y)
    income_x = fs.transform(income_x)
    # transform test input data
    income_x_t = fs.transform(income_x_t)

    income_x_train = income_x
    income_y_train = income_y
    income_x_test = income_x_t
    income_y_test = income_y_t

    return income_x_train, income_y_train, income_x_test, income_y_test


if __name__ == "__main__":
    df = pd.read_csv("training.csv", index_col=0)
    df_test = pd.read_csv("test.csv", index_col=0)
    df, df_test = prepareData(df, df_test)
    # corr = 0
    # k_val = 1
    # for k in range(1, 253):
    #     income_x_train, income_y_train, income_x_test, income_y_test = create_sets(
    #         df, 0, df_test, k)
    #     model = linear_model.LinearRegression()
    #     model.fit(income_x_train, income_y_train)
    #     y_pred = model.predict(income_x_test)

    #     temp, _ = pearsonr(income_y_test, y_pred)
    #     if (temp > corr):
    #         k_val = k
    #         corr = temp

    income_x_train, income_y_train, income_x_test, income_y_test = create_sets(
        df, 1, df_test, 158)
    acc = 0
    k_val1 = 0
    k_vals = [1, 3, 5, 7, 9, 11, 13]
    val = []
    for k in range(1, 14, 2):

        k_model = KNeighborsClassifier(n_neighbors=k)
        k_model.fit(income_x_train, income_y_train)
        y_pred_k = k_model.predict(income_x_test)
        temp = accuracy_score(income_y_test, y_pred_k)
        val.append(temp)
        if (temp > acc):
            acc = temp
            k_val1 = k

        # print('Pearsons correlation: {}'.format(corr))
        # print('Mean Squared Error:', mean_squared_error(
        #     income_y_test, y_pred))
        # print("best k is {}".format(k_val))
    plt.plot(k_vals, val)
    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    print('accuracy is: {}'.format(acc))
    print("best k is {}".format(k_val1))
