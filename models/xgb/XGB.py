from xgboost import XGBClassifier

from models.utils import print_evaluation_metrics
from preprocess.main import load_data, prepare_df_for_learning, get_x_y
from sklearn.model_selection import RandomizedSearchCV


def train_xgb(x_train, x_test, y_train, y_test):
    clf = XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100)
    print('start train model')
    clf = clf.fit(x_train, y_train)
    print('finish to train model')
    test_pred = clf.predict(x_test)
    print('finish to predict test set')
    train_pred = clf.predict(x_train)
    print('finish to predict train set')
    return clf, test_pred, train_pred


def parameter_tuning(x, y, x_test, y_test):
    print('\n\n')
    print('Parameter Tuning: ')
    param_test = {
        'max_depth': range(3, 80),
        'min_child_weight': range(1, 20),
        'learning_rate': [i / 10.0 for i in range(1, 9)],
        'n_estimators': range(10, 200, 10),
        'gamma': [i / 10.0 for i in range(0, 5)],
        'subsample': [i / 10.0 for i in range(2, 10)],
        'colsample_bytree': [i / 10.0 for i in range(2, 10)]
    }
    gsearch1 = RandomizedSearchCV(estimator=XGBClassifier(), param_distributions=param_test, n_iter=100, cv=3,
                                  verbose=2, random_state=42, n_jobs=-1)
    gsearch1.fit(x, y)

    print('\n\nResults: ')
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    y_pred = gsearch1.predict(x_test)
    print_evaluation_metrics("XGB", gsearch1, y_pred, [], [], x_test, [], y_test)


if __name__ == '__main__':
    df = load_data("ctr_dataset_train")
    df = prepare_df_for_learning(df)
    x_train, x_test, y_train, y_test, x_train_resampled, y_train_resampled, x_val, y_val = get_x_y(df)

    print('Train on regular data set:')
    clf, test_pred, train_pred = train_xgb(x_train, x_test, y_train, y_test)
    print_evaluation_metrics("XGB", clf, test_pred, train_pred, x_train, x_test, y_train, y_test)
    print('\n\n')

    print('Train on oversampled data set:')
    oversampled_clf, oversampled_test_pred, oversampled_train_pred = \
        train_xgb(x_train_resampled, x_test, y_train_resampled, y_test)
    print_evaluation_metrics("XGB", oversampled_clf, oversampled_test_pred, oversampled_train_pred, x_train, x_test,
                             y_train, y_test, True)

    parameter_tuning(x_val, y_val, x_test, y_test)
