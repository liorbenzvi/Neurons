import pandas as pd
from xgboost import XGBClassifier
from preprocess.main import prepare_df_for_learning

if __name__ == '__main__':
    # train model:
    df = pd.read_csv("../data/ctr_dataset_train.csv", encoding="UTF-8")
    df["Purchase"] = df["Purchase"].fillna(-1)
    df["Purchase"] = df["Purchase"].astype(int)
    df = prepare_df_for_learning(df)
    y_filter = ['Purchase']
    x_filter = df.columns[~df.columns.isin(y_filter)]
    clf = XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100)
    clf = clf.fit(df[x_filter], df[y_filter])

    # predict test:
    df_test = pd.read_csv("../data/ctr_dataset_test.csv", encoding="UTF-8")
    df_test = prepare_df_for_learning(df_test)
    results_df = pd.DataFrame()
    results_df['results'] = clf.predict(df_test)
    (results_df['results']).to_csv("../data/output_13.txt ", encoding="UTF-8", index=False, header=False)
