import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import pickle
from RFCModel import RFCModel

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from operator import itemgetter


def _unpickle(filename):
    with open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def _pickle(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def clean_df(filename, is_json=True, training=True):

    if not is_json :
        df = pd.read_csv(filename)
    else :
        df = pd.read_json(filename)

    if training:
        #create label column
        fraud = (df['acct_type'] == 'fraudster') | (df['acct_type'] == 'fraudster_event')
    else:
        fraud = None

    object_id = df.pop('object_id')

    # Drop uneeded cols and cols with NaNs
    df.drop(['event_published','has_header','sale_duration','venue_latitude', 'venue_longitude'], axis=1, inplace=True)

    df['venue_missing'] = df.venue_name.isnull()
    df.venue_name.fillna('UNKNOWN', inplace=True)

    # Synthesize some columns
    df['event_duration'] = df.event_end - df.event_start
    df['name_caps'] = df.name.map(lambda x: x.isupper())
    df['org_caps'] = df.org_name.map(lambda x: x.isupper())
    df['venue_caps'] = df.venue_name.map(lambda x: x.isupper())
    df['num_previous_payouts'] = df.previous_payouts.map(lambda x: len(x))

    # Change dates to durations
    uc = df.user_created
    df['approx_payout_date'] = df.approx_payout_date - uc
    df['event_created'] = df.event_created - uc
    df['event_end'] = df.event_end - uc
    # df['event_published'] = df.event_published - uc
    df['event_start'] = df.event_start - uc

    # Get rid of the user_created date.  This might be biased.
    df.drop(['user_created'],axis=1, inplace=True)

    # Only include numeric and boolean columns
    dic_columntypes = df.columns.to_series().groupby(df.dtypes).groups
    cols = []
    for k,v in dic_columntypes.items():
        # print(str(k))
        if str(k) in ['int64', 'float64', 'bool']:
            for item in v:
                cols.append(str(item))

    # Mask out non-numeric, non-bool columns
    df = df[cols]
    # Sort the columns into alpha order, every time
    df.sort_index(axis=1, inplace=True)

    print('Columns:')
    print(df.columns)
    return df, fraud, object_id


class XGBoostModel:

    def __init__(self, use_rfc=True):
        self.use_rfc = use_rfc
        if self.use_rfc:
            # Instantiate Random Forest Classifier
            self.rfc = RFCModel()
            self.rfc.unpickle()

    def load_train_data(self):
        self.df, y, _ = clean_df('data/data.json', training=True)

        if self.use_rfc:
            # Include results from random forest classifier as new column
            rfc_probs = self.rfc.predict_proba_all()
            self.df['rfc_proba'] = rfc_probs

        X = self.df.values

        self.features = self.df.columns
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    def load_test_data(self):
        self.df, _, oid = clean_df('data/data_point.json', training=False)

        if self.use_rfc:
            # Include results from random forest classifier as new column
            rfc_probs = self.rfc.predict_proba('data/data_point.json')
            self.df['rfc_proba'] = rfc_probs

        return self.df.values, oid

    def load_one(self, one_json):
        # with open('one.json', 'w') as f:
        #     temp = '[' + one_json + ']'
        #     f.write(temp)

        self.df, _, oid = clean_df('[' + one_json + ']', training=False)

        if self.use_rfc:
            # Include results from random forest classifier as new column
            rfc_probs = self.rfc.predict_proba('data/data_point.json')
            self.df['rfc_proba'] = rfc_probs

        return self.df.values, oid


    def fit(self):
        self.model = XGBClassifier(max_depth=8,\
                                # reg_alpha=.8,\
                                n_estimators=200,\
                                scale_pos_weight=10.13,\
                                learning_rate=0.1)

        self.model.fit(self.X_train, self.y_train)

    @property
    def feature_importances_(self):
        #I couldn't call the master class, so just copy-n-pasted
        #See https://github.com/dmlc/xgboost/commit/dd477ac903eb6f658d6fb2984763c3f8a4516389#diff-2c197a11c1b576e821f5942be9eab74c
        b = self.model.booster()
        fs = b.get_fscore()
        all_features = [fs.get(f, 0.) for f in b.feature_names]
        all_features = np.array(all_features, dtype=np.float32)
        return all_features / all_features.sum()

    def plot_features(self, save_img_dir=None, img_name_prefix='', ext='svg'):
        '''
        use ext='svg' for web!
        add save_file_dir location to save images
        save_file_dir has NO trailing slash!
        eg 'static/images'
        to keep multiple images saved add prefix string
        prefix will be added to image file name

        '''

        # this is needed to fix lable clipping in saved files
        from matplotlib import rcParams
        rcParams.update({'figure.autolayout': True})

        #severly modified from https://gist.github.com/light94/6c42df29f3232ae31e52
        b = self.model.booster()
        fs = b.get_fscore()
        #print('feature...')
        #print(b.feature_names)
        #all_features = {f:fs.get(f, 0.) for f in b.feature_names}
        #need to add real feature names
        all_features = {self.features[i]:float(fs.get('f'+str(i),0.)) for i in range(len(b.feature_names))}
        importance = sorted(all_features.items(), key=itemgetter(1))

        ff = pd.DataFrame(importance, columns=['feature', 'fscore'])
        ff['fscore'] = ff['fscore'] / ff['fscore'].sum()

        #"plot 1"
        ax = ff.fscore.plot(xticks=ff.index, rot=65)
        ax.set_xticklabels(ff.feature)
        plt.title('XGBoost F-scores by feature')

        if save_img_dir is not None:
            plt.savefig('{}/{}feature_fscores.{}'.format(save_img_dir, img_name_prefix, ext))
        plt.show()

        #"plot 2"
        ff.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        if save_img_dir is not None:
            plt.savefig('{}/{}features_barh.{}'.format(save_img_dir, img_name_prefix, ext))
        plt.show()
        plt.close()


    def pickle(self):
        _pickle(self.model, 'data/XGBoostModel.pkl')

    def unpickle(self):
        self.model = _unpickle('data/XGBoostModel.pkl')

    def score(self):
        y_pred = self.model.predict(self.X_test)
        probs = self.model.predict_proba(self.X_test)[:,1]
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("f1: %.2f" % f1)
        print('Confusion matrix')
        print(np.array([['TN','FP'],['FN', 'TP']]))
        print(confusion_matrix(self.y_test, y_pred))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        prob = self.model.predict_proba(X)
        return prob[:,1]


def write_class_result(df, oid, prob):
    df['object_id'] = oid
    df['prob_fraud'] = prob
    df.to_csv('data/sample.csv')
    # df.to_json('data/sample.json')

if __name__ == '__main__':
    # model = XGBoostModel(use_rfc=True)
    # model.load_train_data()
    # model.fit()
    # model.pickle()
    # model.score()
    # # print(model.feature_importances_)
    # model.plot_features(save_img_dir='.',ext='png')

    model = XGBoostModel(use_rfc=True)
    model.unpickle()
    X, oid = model.load_test_data()  # json input goes here
    print(model.predict(X))
    prob = model.predict_proba(X)
    print(prob)
    # write_class_result(model.df, oid, prob)
