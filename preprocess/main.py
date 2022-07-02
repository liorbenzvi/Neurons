import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt


def load_data(file_name):
    print("Load data from file: " + file_name)
    df = pd.read_csv("../data/" + file_name + ".csv", encoding="UTF-8")
    print(df.head())
    df["Purchase"] = df["Purchase"].fillna(-1)
    df["Purchase"] = df["Purchase"].astype(int)
    return df


def data_exploration(df):
    print('General describe: ')
    print(df.describe())
    print(df.info())
    print('Amount of rows: ' + str(len(df)))
    print('Amount of columns: ' + str(len(df.columns)))
    print('Describe by columns')
    for col in df.columns:
        print('Statistics for column: ' + col)
        if is_numeric_dtype(df[col]):
            print('Average: ' + str(df[col].mean()))
            print('Median: ' + str(df[col].median()))
            print('Standard deviation: ' + str(df[col].std()))
            print('Max value: ' + str(df[col].max()))
            print('Min value: ' + str(df[col].min()))
        print('Amount of distinct values: ' + str(len(pd.unique(df[col]))))
        print('Amount of missing values: ' + str(df[col].isna().sum()))
        print('Count amounts by value: ')
        print(df[col].value_counts())

    print('\nCorrelation matrix: ')
    corr_matrix = df.corr()
    corr_matrix.to_csv("../data/correlation_matrix.csv", encoding="UTF-8")
    print(corr_matrix.to_string())
    plt.imshow(corr_matrix, cmap='hot', interpolation='nearest')
    plt.savefig('correlation_matrix.png')


def prepare_df_for_learning(df):
    df = remove_rows_without_target_value(df)
    return df


def remove_rows_without_target_value(df):
    print('Remove rows without target value: ')
    rows_before = len(df)
    print('Amount of rows in df is: ' + str(rows_before))
    print('Going to remove rows without Purchase value')
    df = df.drop(df[df.Purchase == -1].index)
    rows_after = len(df)
    print('Amount of rows in df is: ' + str(rows_after))
    print('Total ' + str(rows_before - rows_after) + ' rows were removed')
    return df


if __name__ == '__main__':
    df = load_data("ctr_dataset_train")
    data_exploration(df)
    print('\n\n')
    df = prepare_df_for_learning(df)
