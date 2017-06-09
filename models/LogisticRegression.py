import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import process_data as proc
#import roc
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.preprocessing import StandardScaler


def main():
    data = proc.read_data()
    features, yfill = proc.features_yfill(data)
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=42, stratify =yfill)
    X_train_over, y_train_over = proc.oversample(X_train,y_train, r = 0.3)
    #plot_roc(X_train, y_train, 'LogisticRegression', LogisticRegression(C=1e5,penalty='l2'))
    '''
    model_over = runLR(X_train_over, X_test, y_train_over, y_test)
    test_results(model_over, X_test, y_test)
    '''

    model = runLR(X_train.values, X_test, y_train.values, y_test)
    test_results(model, X_test, y_test)

def runLR(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(C=1e5,penalty='l2')
    clf.fit(X_train,y_train)
    return clf

def test_results(model, X_test, y_test):
    y_predict = model.predict(X_test)
    print("score (accuracy):", model.score(X_test, y_test))
    print("precision:", precision_score(y_test, y_predict))
    print("recall:", recall_score(y_test, y_predict))
    print(np.array([['TN','FP'],['FN', 'TP']]))
    print(confusion_matrix(y_test, y_predict))


def plot_roc(X, y, name, estimator):
    scaler = StandardScaler()
    # investigate ouput from scaler
    X = scaler.fit_transform(X)
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = estimator
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        print(fpr)
        print(tpr)
        print(thresholds)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classification of fragment library to MTH1 using {}'.format(name))
    plt.legend(loc="lower right")
    plt.savefig('ROC_{}.png'.format(name))
    plt.close()




if __name__ == '__main__':
    main()
    '''
    data = proc.read_data()
    features, yfill = proc.features_yfill(data)
    X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=42, stratify =yfill)
    X_train_over, y_train_over = proc.oversample(X_train,y_train, r = 0.3)
    #plot_roc(X_train.values, y_train.values, 'LogisticRegression', LogisticRegression(C=1e5,penalty='l2'))
    plot_roc(X_train_over, y_train_over, 'LogisticRegression_oversampling', LogisticRegression(C=1e5,penalty='l2'))
    '''
