import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, mean_squared_error, recall_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import shuffle
from scipy.stats import pearsonr
import sys


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

    for val in new_list:
        if val in list1:
            df = df.drop([val], axis=1)
        if val in list2:
            df_test = df_test.drop([val], axis=1)

    for column in df.columns:

        imputer = SimpleImputer(missing_values=np.nan,
                                strategy="median")
        df[column] = imputer.fit_transform(df[[column]])
    for column in df_test.columns:
        imputer = SimpleImputer(missing_values=np.nan,
                                strategy="median")
        df_test[column] = imputer.fit_transform(df_test[[column]])
    df = shuffle(df)
    df_test = shuffle(df_test)
    return df, df_test


def create_sets(df, algo, df_test):

    if algo == 0:
        income_x = df.drop(
            ['AMT_INCOME_TOTAL'], axis=1).values
        income_y = df['AMT_INCOME_TOTAL'].values
        income_x_t = df_test.drop(
            ['AMT_INCOME_TOTAL'], axis=1).values
        income_y_t = df_test['AMT_INCOME_TOTAL'].values
        income_x_t_id = df_test.index
    else:
        income_x = df.drop(
            ['TARGET'], axis=1).values
        income_y = df['TARGET'].values
        income_x_t = df_test.drop(
            ['TARGET'], axis=1).values
        income_y_t = df_test['TARGET'].values
        income_x_t_id = df_test.index

    fs = SelectKBest(score_func=f_regression, k=158)
    fs.fit(income_x, income_y)
    income_x = fs.transform(income_x)
    income_x_t = fs.transform(income_x_t)

    income_x_train = income_x
    income_y_train = income_y
    income_x_test = income_x_t
    income_y_test = income_y_t

    return income_x_train, income_y_train, income_x_test, income_y_test, income_x_t_id


if __name__ == "__main__":
    training_p = sys.argv[1]
    test_p = sys.argv[2]
    df = pd.read_csv(training_p, index_col=0)
    df_test = pd.read_csv(test_p, index_col=0)
    df, df_test = prepareData(df, df_test)
    income_x_train, income_y_train, income_x_test, income_y_test, id_col = create_sets(
        df, 0, df_test)
    model = linear_model.LinearRegression()
    model.fit(income_x_train, income_y_train)
    y_pred = model.predict(income_x_test)

    ans = pd.DataFrame(y_pred)
    ans.index.names = ['SK_ID_CURR']
    ans.columns = ['predicted_income']
    ans['predicted_income'] = ans['predicted_income'].apply(lambda x: int(x))

    corr, _ = pearsonr(income_y_test, y_pred)

    # print('Pearsons correlation: {}'.format(corr))
    # print('Mean Squared Error:', mean_squared_error(
    #     income_y_test, y_pred))
    p1summ = pd.DataFrame({'zid': ["z5309451"], 'MSE': ["{:0.2f}".format(mean_squared_error(
        income_y_test, y_pred))], "correlation": ["{:0.2f}".format(corr)]})
    p1summ.set_index('zid', inplace=True)
    p1summ.to_csv("z5309451.PART1.summary.csv")
    ans.to_csv("z5309451.PART1.output.csv")

    income_x_train, income_y_train, income_x_test, income_y_test, id_col1 = create_sets(
        df, 1, df_test)
    k_model = KNeighborsClassifier(n_neighbors=13)
    k_model.fit(income_x_train, income_y_train)
    y_pred_k = k_model.predict(income_x_test)
    # print("accuracy:\t", accuracy_score(income_y_test, y_pred_k))
    # print("precision:\t", precision_score(
    #     income_y_test, y_pred_k, average=None))
    p2summ = pd.DataFrame({'zid': ["z5309451"], 'average_precision': ["{:0.2f}".format(average_precision_score(
        income_y_test, y_pred_k))], "average_recall": ["{:0.2f}".format(recall_score(income_y_test, y_pred_k, average='macro', zero_division=1))], "accuracy": ["{:0.2f}".format(accuracy_score(income_y_test, y_pred_k))]})
    p2summ.set_index('zid', inplace=True)
    p2summ.to_csv("z5309451.PART2.summary.csv")

    ans = pd.DataFrame(y_pred_k)
    ans.index.names = ['SK_ID_CURR']
    ans.columns = ['predicted_target']
    ans['predicted_target'] = ans['predicted_target'].apply(lambda x: int(x))
    ans.to_csv("z5309451.PART2.output.csv")
