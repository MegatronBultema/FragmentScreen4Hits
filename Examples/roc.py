import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import process_data as proc
plt.close()

def plot_roc(X, y, name, clf_class, **kwargs):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
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
    data = proc.read_data()
    features, yfill = proc.features_yfill(data)
    features_over, yfill_over = proc.oversample(features, yfill, r = 0.3)

    #RandomForestClassifier(class_weight = cls_w, n_estimators = num_est)
    plot_roc(features, yfill, 'RandomForestClassifier', RandomForestClassifier,  max_depth = 10,
     max_features= 30, min_samples_leaf= 2, min_samples_split = 2)
    plot_roc(features_over, yfill_over, 'RandomForestClassifier_oversample', RandomForestClassifier,  max_depth = 10, max_features= 30, min_samples_leaf= 2, min_samples_split = 2)
