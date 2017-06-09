import pandas as pd
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import cross_val_predict
import process_data as proc
#import roc
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp



def define_model2():
    # baseline model
	#num_neurons_in_layer = 100
	model = Sequential()
	model.add(Dense(output_dim=100, input_dim=1130, kernel_initializer='normal', activation='relu'))
	model.add(Dense(output_dim=100, kernel_initializer='normal', activation='relu'))
	model.add(Dense(output_dim=50, kernel_initializer='normal', activation='relu'))
	model.add(Dense(output_dim=1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

#https://vkolachalama.blogspot.com/2016/05/keras-implementation-of-mlp-neural.html
def generate_results(y_test, y_score):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.savefig('ROC_KerasClassifier.png')
    print('AUC: %f' % roc_auc)

def plot_rocNN(X, y, name):
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

def test_results(model, X_test, y_test):
	y_predict = model.predict(X_test)
	print("score (accuracy):", model.score(X_test, y_test))
	print("precision:", precision_score(y_test, y_predict))
	print("recall:", recall_score(y_test, y_predict))
	print(np.array([['TN','FP'],['FN', 'TP']]))
	print(confusion_matrix(y_test, y_predict))


if __name__ == '__main__':
	data = proc.read_data()
	features, yfill = proc.features_yfill(data)
	X_train, X_test, y_train, y_test = train_test_split(features, yfill, test_size=0.20, random_state=2, stratify =yfill)
	rng_seed = 2 # set random number generator seed
	np.random.seed(rng_seed)
	X_train_over, y_train_over = proc.oversample(X_train,y_train, r = 0.3)
	#print test results
	estimator = KerasClassifier(build_fn=define_model2, nb_epoch=300, verbose=1, batch_size =30)

	#estimator.fit(X_train_over, y_train_over)
	#test_results(estimator, X_test.values, y_test.values)


	estimator.fit(X_train.values, y_train.values)
	pprob = estimator.predict_proba(X_test.values)
	pdf = pd.DataFrame(pprob)
	print(pdf)
	pdf['myH'] = pdf[1].map(lambda x: 1 if x>1 else 0)
	my_pred = pdf['myH'].values
	print(my_pred)
	print("precision:", precision_score(y_test, my_pred))
	print("recall:", recall_score(y_test, my_pred))
	print(np.array([['TN','FP'],['FN', 'TP']]))
	print(confusion_matrix(y_test, my_pred))
	#test_results(estimator, X_test.values, y_test.values)

	# evaluate model with standardized dataset
	'''
	estimator = KerasClassifier(build_fn=define_model2, nb_epoch=100, verbose=0, batch_size =30)
	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	results = cross_val_score(estimator, X_train.values, y_train.values, cv=kfold)
	predicted = cross_val_predict(estimator, X_train.values, y_train.values, cv=kfold)
	print("Results all features: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("precision:", precision_score(y_train, predicted))
	print("recall:", recall_score(y_train, predicted))
	print(np.array([['TN','FN'],['FP', 'TP']]))
	print(confusion_matrix(y_train, predicted))
	'''

	# evaluate model with the oversampled dataset
	'''
	estimator = KerasClassifier(build_fn=define_model2, nb_epoch=100, verbose=0, batch_size =30)
	kfold = StratifiedKFold(n_splits=5, shuffle=True)
	results_over = cross_val_score(estimator, X_train_over, y_train_over, cv=kfold)
	predicted_over = cross_val_predict(estimator, X_train_over, y_train_over, cv=kfold)
	print("Results all features oversampled: %.2f%% (%.2f%%)" % (results_over.mean()*100, results_over.std()*100))
	print("precision:", precision_score(y_train_over, predicted_over))
	print("recall:", recall_score(y_train_over, predicted_over))
	print(np.array([['TN','FN'],['FP', 'TP']]))
	print(confusion_matrix(y_train_over, predicted_over))
	'''

	# run plot ROC curve
	'''
	estimator = KerasClassifier(build_fn=define_model2, nb_epoch=100, verbose=0, batch_size =30)
	features_over, yfill_over = proc.oversample(features,yfill, r = 0.3)
	plot_rocNN(features_over, yfill_over, 'NeuralNet_oversample')
	'''



	#y_score = model.predict(X_test)
	#generate_results(y_test.values, y_score.values)

	'''
	model = define_model2()
	model.fit(X_train.values, y_train.values, validation_data = (X_test.values, y_test.values), epochs=2, verbose =2 )
	'''


	#run again with only bit features
	'''
	bits, ybits = features_yfill(data)
	X_trainb, X_testb, y_trainb, y_testb = train_test_split(bits, ybits, test_size=0.20, random_state=42, stratify =ybits)
	rng_seed = 2 # set random number generator seed
	np.random.seed(rng_seed)
	'''

	'''
	estimator_bits = KerasClassifier(build_fn=define_model2, nb_epoch=100, verbose=0, batch_size =30)
	kfold_bits = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng_seed)
	results_bits = cross_val_score(estimator, X_trainb.values, y_trainb.values, cv=kfold)
	predicted_bits = cross_val_predict(estimator, X_trainb.values, y_trainb.values, cv=kfold)
	print("Results only bits data: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
	print("precision:", precision_score(y_train, predicted_bits))
	print("recall:", recall_score(y_train, predicted_bits))
	print(np.array([['TN','FN'],['FP', 'TP']]))
	print(confusion_matrix(y_train, predicted_bits))
	'''
