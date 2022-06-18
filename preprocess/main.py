import pandas as pd


def load_data(file_name):
    print("load data from file: " + file_name)
    df = pd.read_csv("../data/" + file_name + ".csv", encoding="UTF-8")
    print(df.head())
    df["Purchase"] = df["Purchase"].fillna(-1)
    df["Purchase"] = df["Purchase"].astype(int)
    return df


def data_exploration(df):
    print('General describe: ')
    print(df.describe())
    print(df.info())
    print(df.summery())
    print('Amount of rows: ' + str(len(df.rows)))
    print('Amount of columns: ' + str(len(df.columns)))
    print('Describe by columns')
    for col in df.columns:
        print('statistics for column: ' + col)
        print('Average: ' + str(df[col].mean()))
        print('Median: ' + str(df[col].median()))
        print('Standard deviation: ' + str(df[col].std()))
        print('Max value: ' + str(df[col].max()))
        print('Min value: ' + str(df[col].min()))

        print('Amount of distinct values: ' + str(len(pd.unique(df[col]))))
        print('Amount of missing values: ' + str(df[col].isna().sum()))
        print('Count amounts by value: ')
        print(df[col].value_counts())


if __name__ == '__main__':
    df = load_data("ctr_dataset_train")
    data_exploration(df)
