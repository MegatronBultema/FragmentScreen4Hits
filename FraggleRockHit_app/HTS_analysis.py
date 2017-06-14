import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import process_data as proc
import heapq
from sets import Set
from sklearn.metrics import roc_curve

def set_threshold_recall(model, X,y):
    '''
    This function is only useful when the HTS screen has been scored.
    In that case it can be used to set a threshold and review the preformance of the model.
    Set a threshold for (median of) maximal recall.
        The function iterates through 100 possible threshold values (0 to 1.0) and records the recall and precision for each threshold. Then it finds the unique values in the recall list and selects the index values for the recall value closest to 1.0 (but not 1.0). The median threshold from the list that would return the same recall is returned along with the stats for that threshold.
    Input
        model -  trained on fragment library.
        X - features
        y -scored hits
    Output
        precision - at selected threshold
        recall- at selected threshold
        median_recall - list of index for median recall (same as index for corresponding threshold), 1 value if odd number of values in best_recall_index or 2 if even
        threshold[median_recall[0]] - threshold cooresponding to the median recall
    '''
    precision = []
    recall = []
    f1score = []
    threshold = list(np.linspace(0.0,1.0,100))
    for i in threshold:
    	pprob = model.predict_proba(X)
    	pdf = pd.DataFrame(pprob)
    	pdf['myH'] = pdf[1].map(lambda x: 1 if x>i else 0)
    	my_pred = pdf['myH'].values
    	precision.append(precision_score(y, my_pred))
    	recall.append(recall_score(y, my_pred))
    	f1score.append(f1_score(y, my_pred))

    recall_set = list(Set(recall))
    recall_set.sort()
    best_recall= recall_set[-2]
    best_recall_index = [i for i,v in enumerate(recall) if v == best_recall]
    median_recall = [best_recall_index[i] for i in range((len(best_recall_index)/2) - (1 if float(len(best_recall_index)) % 2 == 0 else 0), len(best_recall_index)/2+1)]
    bestprecision_recall = best_recall_index[-1]
    return precision, recall, median_recall, threshold[median_recall[0]]

def set_threshold_precision(model, X, y):
    '''
    This function is only useful when the HTS screen has been scored.
    In that case it can be used to set a threshold and review the preformance of the model.
    Set a threshold for (median of) maximal precision.
        The function iterates through 100 possible threshold values (0 to 1.0) and records the recall and precision for each threshold. Then it finds the unique values in the recall list and selects the index values for the recall value closest to 1.0 (but not 1.0). The median threshold from the list that would return the same precision is returned along with the stats for that threshold.
    Input
        model -  trained on fragment library.
        X - features
        y -scored hits
    Output
        precision - at selected threshold
        recall- at selected threshold
        median_precision - list of index for median precision (same as index for corresponding threshold), 1 value if odd number of values in best_precision_index or 2 if even
        threshold[median_precision[0]] - threshold cooresponding to the median precision
    '''
    precision = []
    recall = []
    f1score = []
    threshold = list(np.linspace(0.0,1.0,100))
    for i in threshold:
    	pprob = model.predict_proba(X)
    	pdf = pd.DataFrame(pprob)
    	pdf['myH'] = pdf[1].map(lambda x: 1 if x>i else 0)
    	my_pred = pdf['myH'].values
    	precision.append(precision_score(y, my_pred))
    	recall.append(recall_score(y, my_pred))
    	f1score.append(f1_score(y, my_pred))
    precision_set = list(Set(precision))
    precision_set.sort()
    sugg_precision= precision_set[-4]
    sugg_precision_index = [i for i,v in enumerate(precision) if v == sugg_precision]
    median_precision = [sugg_precision_index[i] for i in range((len(sugg_precision_index)/2) - (1 if float(len(sugg_precision_index)) % 2 == 0 else 0), len(sugg_precision_index)/2+1)]
    bestrecall_precision = sugg_precision_index[-1]
    return precision, recall, median_precision, threshold[median_precision[0]]

def print_threshold(model, X, y, threshold):
    '''
    Get stats for a trained model at a given threshold.
    Input
        model - trained with fragment (or equivilant) data
        X - features
        y - scored hits
        threshold - value above which a given probability is scored a hit
    Output
        precision
        recall
        fpr - false positiv Rate
        fpr_test - list of fpr for given thresholds
        tpr_test - list of tpr for given thresholds
        cm - confusion matrix
    '''
    pprob = model.predict_proba(X)
    pdf = pd.DataFrame(pprob)
    #print(pdf)
    pdf['myH'] = pdf[1].map(lambda x: 1 if x>threshold else 0)
    my_pred = pdf['myH'].values
    precision =  precision_score(y, my_pred)
    recall = recall_score(y, my_pred)
    fpr_test, tpr_test, thresholds_test = roc_curve(y, pprob[:,1])
    #print(np.array([['TN','FP'],['FN', 'TP']]))
    cm = confusion_matrix(y, my_pred)
    fpr = cm[0][1]/float(cm[0][1] + cm[0][0])
    return precision, recall, fpr, fpr_test, tpr_test, cm


def score_HTS(model, hts_features, ids):
    '''
    This function will be used for the general HTS data uploaded with out previous scored hits.
    Input
        model - trained with fragment library
        hts_features - features of HTS scree, MUST BE THE SAME DIMENSIONS AS TRAINING data
        ids -
    Return
        sorted_pprob - dataframe with ids sorted by probability of being a hit
    '''
    pprob = pd.DataFrame()
    pprob['ID']= ids
    pprob['Hit_Probability'] = pd.DataFrame(model.predict_proba(hts_features)[:,1])
    #pprob['AMW']= hts_features['AMW']
    sorted_pprob = pprob.sort_values('Hit_Probability', ascending = False)
    return sorted_pprob

def feature_importance(bits, rf):
    '''
    I don't think this is functional
    '''
    features = bits.columns
    importances = rf.feature_importances_
    fi_idx = np.argsort(rf.feature_importances_)[::-1]
    print("\n top five features by importance:", list(bits.columns[fi_idx[:5]]))
    print("\n top five importance scores :", list(importances[fi_idx[:5]]))

def plot_features(features, model, name, n=10):
    '''
    Plot n number of most important features for a trained model
    Input
        features - feature space, must be the same between training and HTS data, only used to get column names
        model - trained on fragment (or equivilant) data
        name - string to save under
        n - number of features to display
    '''
    feature_names = features.columns
    importances = model.feature_importances_
    fi_idx = np.argsort(model.feature_importances_)[::-1]
    topn_idx = fi_idx[:n]
    #std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(10), importances[topn_idx], color="r", align="center")
    plt.xticks(range(10), features[topn_idx], rotation=90)
    plt.xlim([-1, 10])
    plt.tight_layout()
    #plt.ylim([0.01, 0.1])
    plt.savefig('static/Feature_Importance_{}.png'.format(name))
    plt.close()


if __name__ == '__main__':
    pass
