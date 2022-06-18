import pandas as pd

def load_data(file_name):
    print("load data from file: " + file_name)
    df = pd.read_csv("../data/" + file_name + ".csv", encoding="UTF-8")
    print(df.head())
    df["Purchase"] = df["Purchase"].astype(int)
    return df



if __name__ == '__main__':
    df = load_data("ctr_dataset_train")