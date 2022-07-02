import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, recall_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from preprocess.main import load_data, prepare_df_for_learning, get_x_y
from sklearn.metrics import precision_score
from sklearn import metrics
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


def calc_acc(pred, actual):
    total = len(pred)
    correct = len([i for i, j in zip(pred, actual.values) if i == j])
    return correct, total


def print_accuracy(test_pred, y_test, train_pred, y_train):
    print('Accuracy on Test Set:')
    correct, total = calc_acc(test_pred, y_test)
    print("Accuracy is: {0} %".format(str(round((correct / total) * 100, 2))))
    print('Accuracy on Train Set:')
    correct, total = calc_acc(train_pred, y_train)
    print("Accuracy is: {0} %".format(str(round((correct / total) * 100, 2))))


def print_confusion_matrix(clf, x, y, is_oversampled=False):
    print('Confusion matrix: ')
    disp = ConfusionMatrixDisplay.from_estimator(clf, x, y, cmap=plt.cm.Blues)
    disp.ax_.set_title('Confusion matrix')
    print(disp.confusion_matrix)
    file_name = 'oversampled_confusion_matrix' if is_oversampled else 'confusion_matrix'
    plt.savefig('png_files/' + file_name + '.png', dpi=150)
    plt.clf()


def print_business_value(y_true, y_pred):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = len([y_pred_i for y_pred_i, y_true_j in zip(y_pred, y_true.values) if y_pred_i == 1 and y_true_j == 1])
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP =  len([y_pred_i for y_pred_i, y_true_j in zip(y_pred, y_true.values) if y_pred_i == 1 and y_true_j == 0])
    print('Total business value from model is: ' + str(TP * 420 - FP * 200))


def print_evaluation_metrics(clf, test_pred, train_pred, x_train, x_test, y_train, y_test, is_oversampled=False):
    print('\n\n')
    print('Model evaluation_metrics: ')
    print_accuracy(test_pred, y_test, train_pred, y_train)
    print_confusion_matrix(clf, x_test, y_test, is_oversampled)
    print('Precision score is: ' + str(precision_score(y_test, test_pred)))
    print('Recall score is: ' + str(recall_score(y_test, test_pred)))
    print_auc_plt(clf, x_test, y_test, is_oversampled)
    print_business_value(y_test, test_pred)


def print_auc_plt(clf, x_test, y_test, is_oversampled):
    y_pred_proba = clf.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    file_name = 'oversampled_auc_plt' if is_oversampled else 'auc_plt'
    plt.savefig('png_files/' + file_name + '.png', dpi=150)
    plt.clf()


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
    print_evaluation_metrics(gsearch1, y_pred, [], x, x_test, y, y_test)


if __name__ == '__main__':
    df = load_data("ctr_dataset_train")
    df = prepare_df_for_learning(df)
    x_train, x_test, y_train, y_test, x_train_resampled, y_train_resampled, x_val, y_val = get_x_y(df)

    print('Train on regular data set:')
    clf, test_pred, train_pred = train_xgb(x_train, x_test, y_train, y_test)
    print_evaluation_metrics(clf, test_pred, train_pred, x_train, x_test, y_train, y_test)
    print('\n\n')
    print('Train on oversampled data set:')
    oversampled_clf, oversampled_test_pred, oversampled_train_pred = \
        train_xgb(x_train_resampled, x_test, y_train_resampled, y_test)
    print_evaluation_metrics(oversampled_clf, oversampled_test_pred, oversampled_train_pred, x_train, x_test,
                             y_train, y_test, True)

    parameter_tuning(x_val, y_val, x_test, y_test)


