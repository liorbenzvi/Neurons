from collections import Counter

import numpy
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from statistics import median


def load_data(file_name):
    print("Load data from file: " + file_name)
    df = pd.read_csv("../../data/" + file_name + ".csv", encoding="UTF-8")
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
    print('\n\n')


def oversampling_with_smote(X, y):
    X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    return X_resampled, y_resampled


def remove_nans_from_list(X):
    return [x for x in X if numpy.isnan(x) == False]


def fill_with_distribution(df, column_name):
    column_without_nans = remove_nans_from_list(df[column_name])
    num_column_without_nans = len(column_without_nans)
    num_of_nans = df[column_name].isna().sum()
    count_dict = Counter(column_without_nans)
    values = count_dict.keys()
    probs = [x / num_column_without_nans for x in count_dict.values()]
    values_to_fill = numpy.random.choice(list(values), size=num_of_nans, p=probs)
    df.loc[df[column_name].isnull(), column_name] = values_to_fill


def handle_missing_values(df):
    df['Gender'] = df['Gender'].fillna(df['Gender'].value_counts().idxmax())
    df['Date'] = df['Date'].fillna(df['Date'].value_counts().idxmax())
    df['Credit_score'] = df['Credit_score'].fillna(df['Credit_score'].value_counts().idxmax())

    df['Insurance_risk_score  '] = df['Insurance_risk_score  '].fillna(
        median(remove_nans_from_list(df['Insurance_risk_score  '])))
    df['Garden_m2'] = df['Garden_m2'].fillna(median(remove_nans_from_list(remove_nans_from_list(df['Garden_m2']))))
    df['Living_room_m2'] = df['Living_room_m2'].fillna(
        median(remove_nans_from_list(remove_nans_from_list(df['Living_room_m2']))))
    df['Bedrooms_ m2'] = df['Bedrooms_ m2'].fillna(
        median(remove_nans_from_list(remove_nans_from_list(df['Bedrooms_ m2']))))
    df['Home_age'] = df['Home_age'].fillna(median(remove_nans_from_list(remove_nans_from_list(df['Home_age']))))
    df['Home_evaluation'] = df['Home_evaluation'].fillna(median(remove_nans_from_list(df['Home_evaluation'])))
    df['Paid_life_premium '] = df['Paid_life_premium '].fillna(
        numpy.average(remove_nans_from_list(df['Paid_life_premium '])))
    df['Offered_dwelling insurance'] = df['Offered_dwelling insurance'].fillna(
        median(remove_nans_from_list(df['Offered_dwelling insurance'])))
    fill_with_distribution(df, 'Floor')
    fill_with_distribution(df, 'Num_residents_floor')
    df.dropna()
    return df


def add_new_features(df):
    df = add_date_features(df)
    df['is_private'] = (df['Floor'] == 0.0).astype(int)
    df['is_alone_in_floor'] = (df['Num_residents_floor'] == 1.0).astype(int)
    return df


def add_date_features(df):
    df[['day', 'month', 'year']] = df['Date'].str.split('/', expand=True)
    df['year'] = df['year'].fillna(-1)
    df['year'] = df['year'].astype(int)
    df['years_until_today'] = 2022 - df['year']
    df['years_until_today'] = df['years_until_today'].replace(2023, -1)
    df = df.drop(columns=['day', 'Date'])
    return df


def prepare_df_for_learning(df):
    df = remove_rows_without_target_value(df)
    df = handle_missing_values(df)
    df = add_new_features(df)
    df = label_encoding(df)
    df = norm_data(df)
    return df


def label_encoding(df):
    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            df[col] = df[col].astype('category').cat.codes
    return df


def one_hot_encode(df, column):
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = df.drop(column, axis=1)
    df = df.join(one_hot)
    return df

def plot_hist(df, column):
    plt.hist(df[column], density=True, bins=30)  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel(column);
    plt.show()

def norm_data(df):
   # for column in df.columns:
   #     plot_hist(df, column)
    df = one_hot_encode(df, "City")
    df = one_hot_encode(df, "Insurance_district")
    if "Unnamed: 0" in df.columns:
        unnamed = df["Unnamed: 0"]
        df = df.drop("Unnamed: 0", axis=1)
    id = df["User_ID"]
    df = df.drop("User_ID", axis=1)
    normalized_df = (df - df.min()) / (df.max() - df.min())
    if "Unnamed: 0" in df.columns:
        normalized_df["Unnamed: 0"] = unnamed
    normalized_df["User_ID"] = id
    return normalized_df


def remove_rows_without_target_value(df):
    print('Remove rows without target value: ')
    rows_before = len(df)
    print('Amount of rows in df is: ' + str(rows_before))
    if 'Purchase' in df.columns:
        print('Going to remove rows without Purchase value')
        df = df.drop(df[df.Purchase == -1].index)
    rows_after = len(df)
    print('Amount of rows in df is: ' + str(rows_after))
    print('Total ' + str(rows_before - rows_after) + ' rows were removed')
    return df


def get_x_y(df):
    y_filter = ['Purchase']
    x_filter = df.columns[~df.columns.isin(y_filter)]
    x_train, x_test, y_train, y_test = train_test_split(df[x_filter], df[y_filter], test_size=0.2, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=8)
    x_train_resampled, y_train_resampled = oversampling_with_smote(x_train, y_train)
    print_oversampling_info(x_train, x_train_resampled)
    return x_train, x_test, y_train, y_test, x_train_resampled, y_train_resampled, x_val, y_val


def print_oversampling_info(x_train, x_train_resampled):
    print('\n\n')
    print('Oversampling with smote: ')
    train_rows = len(x_train)
    resampled_train_rows = len(x_train_resampled)
    print('Amount of rows in regular training set is: ' + str(train_rows))
    print('Amount of rows in oversampled training set is: ' + str(resampled_train_rows))
    print('Added ' + str(resampled_train_rows - train_rows) + ' synthetic rows')


if __name__ == '__main__':
    df = load_data("ctr_dataset_train")
    data_exploration(df)
    df = prepare_df_for_learning(df)
