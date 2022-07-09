from sklearn.metrics import precision_score
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, recall_score


def calc_acc(pred, actual):
    total = len(pred)
    correct = len([i for i, j in zip(pred, actual.values) if i == j])
    return correct, total


def print_accuracy(test_pred, y_test, train_pred, y_train):
    print('Accuracy on Test Set:')
    correct, total = calc_acc(test_pred, y_test)
    print("Accuracy is: {0} %".format(str(round((correct / total) * 100, 2))))
    if len(train_pred) != 0:
        print('Accuracy on Train Set:')
        correct, total = calc_acc(train_pred, y_train)
        print("Accuracy is: {0} %".format(str(round((correct / total) * 100, 2))))


def print_confusion_matrix(model_name, clf, x, y, is_oversampled, is_tuned):
    print('Confusion matrix: ')
    file_name = get_name(is_oversampled, is_tuned, model_name)
    disp = ConfusionMatrixDisplay.from_estimator(clf, x, y, cmap=plt.cm.Blues)
    disp.ax_.set_title(file_name)
    print(disp.confusion_matrix)
    plt.savefig(file_name + '.png', dpi=150)
    plt.clf()


def get_name(is_oversampled, is_tuned, model_name):
    file_name = 'oversampled_confusion_matrix' if is_oversampled else 'confusion_matrix'
    file_name = "tuned_" + file_name if is_tuned else file_name
    file_name = model_name + "_" + file_name
    return file_name


def print_business_value(y_true, y_pred):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = len([y_pred_i for y_pred_i, y_true_j in zip(y_pred, y_true.values) if y_pred_i == 1 and y_true_j == 1])
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = len([y_pred_i for y_pred_i, y_true_j in zip(y_pred, y_true.values) if y_pred_i == 1 and y_true_j == 0])
    print('Total business value from model is: ' + str(TP * 420 - FP * 200))


def print_auc_plt(name, clf, x_test, y_test, is_oversampled, is_tuned):
    y_pred_proba = clf.predict_proba(x_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    file_name = get_auc_file_name(is_oversampled, is_tuned, name)
    plt.savefig(file_name + '.png', dpi=150)
    plt.clf()


def get_auc_file_name(is_oversampled, is_tuned, name):
    file_name = 'oversampled_auc_plt' if is_oversampled else 'auc_plt'
    file_name = "tuned_" + file_name if is_tuned else file_name
    file_name = name + "_" + file_name
    return file_name


def print_evaluation_metrics(model_name, clf, test_pred, train_pred, x_train, x_test, y_train, y_test,
                             is_oversampled=False):
    print('\n\n')
    print('Model ' + model_name + ' evaluation_metrics: ')
    is_tuned = len(train_pred) == 0
    print_accuracy(test_pred, y_test, train_pred, y_train)
    print_confusion_matrix(model_name, clf, x_test, y_test, is_oversampled, is_tuned)
    print('Precision score is: ' + str(precision_score(y_test, test_pred)))
    print('Recall score is: ' + str(recall_score(y_test, test_pred)))
    print_auc_plt(model_name, clf, x_test, y_test, is_oversampled, is_tuned)
    print_business_value(y_test, test_pred)
