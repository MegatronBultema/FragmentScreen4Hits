from flask import Flask, render_template, request, jsonify, redirect, url_for
from data import Articles
import pandas as pd
import numpy as np
import re
#from XGBoostModel import *
import RandomForest_webapp as rf
import HTS_analysis as hts_a
import process_data as proc
import roc
from sklearn.ensemble import RandomForestClassifier
from flask import send_from_directory


#!!! pip install Flask-PyMongo
from flask_pymongo import PyMongo
from pymongo import MongoClient
#import urllib.request, json
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS2 = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
mongo = PyMongo(app)
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS2

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/register', methods=['POST'])
def register():
    return render_template('home.html')

@app.route('/fraggle', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    #Return frggle.html if no file has yet been uploaded
    return render_template('fraggle.html')



@app.route('/uploads/<filename>',methods=['GET', 'POST'])
def uploaded_file(filename):
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_HTS', filename = filename))
    elif request.method == 'GET':
        # make a pd.dataframe of training data
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # use all features and yfill (no NaNs, filled with 0)
        bits, yfill = proc.bits_yfill(df)
        #train test split at 20%
        X_train, X_test, y_train, y_test = rf.train_test_split(bits, yfill, test_size=0.20, random_state=1, stratify =yfill)

        #Optional: oversampling of minority class for training purposes
        #X_train_over, y_train_over = proc.oversample(X_train,y_train, r = 0.3)
        #rffit, y_predict = rf.randomforest(X_train_over, X_test, y_train_over, y_test, num_est=50, cls_w = 'balanced_subsample')

        #fit the Random Forest classifier: would like to add in a grid search
        rffit, y_predict = rf.randomforest(X_train.values, X_test, y_train.values, y_test, num_est=50, cls_w = 'balanced_subsample')

        #pickle the fit model for use with test data
        proc._pickle(rffit, 'RFC_fit.pkl')

        #set_threshold_recall is a function which determines the threshold to set such that recall is optimized (the median of the available thresholds that return the second best recall (not 1.0))
        precision_list, recall_list, median_recall_index, medianrecall_threshold = rf.set_threshold_recall(rffit, X_train, X_test, y_train, y_test)

        #print_threshold uses the trained model and the selected threshold (in this case recall optimized) to return listed statistics
        precision, recall, fpr, fpr_test, tpr_test, cm = rf.print_threshold(rffit, X_train, X_test, y_train, y_test, medianrecall_threshold)
        r_cm = pd.DataFrame(cm)
        proc._pickle(medianrecall_threshold, 'medianrecall_threshold.pkl')

        #make a pd.dataframe of the stats for display
        recall_opt_stats = pd.DataFrame([[format(medianrecall_threshold, '.2f'),format(recall, '.2f'), format(fpr, '.2f'), format(precision, '.2f'), ]], columns = ['Suggested Threshold','True Positive Rate (Recall)', 'False Positive Rate (Fall-out)','Precision'])

        # repeat the threshold selection process for precision optimization
        p_precision, p_recall, p_median_precision, threshold_precision = rf.set_threshold_precision(rffit, X_train, X_test, y_train, y_test)
        p_precision, p_recall, p_fpr, p_fpr_test, p_tpr_test, p_cm = rf.print_threshold(rffit, X_train, X_test, y_train, y_test, threshold_precision)
        p_cm = pd.DataFrame(p_cm)
        precision_opt_stats = pd.DataFrame([[format(threshold_precision, '.2f'),format(p_recall, '.2f'), format(p_fpr, '.2f'), format(p_precision, '.2f'), ]], columns = ['Suggested Threshold','True Positive Rate (Recall)', 'False Positive Rate (Fall-out)','Precision'])

        #produce a ROC plot
        test_prob = rffit.predict_proba(X_test)
        roc.plot_roc(X_train.values, y_train.values, y_test, test_prob, 'Test', RandomForestClassifier,  max_depth = 10, max_features= 30, min_samples_leaf= 2, min_samples_split = 2)
        rf.plot_features(bits, rffit, 'Identifier',n=10)
        #option for overfit data
        #roc.plot_roc(X_train_over, y_train_over, y_test, test_prob, 'Test', RandomForestClassifier,  max_depth = 10, max_features= 30, min_samples_leaf= 2, min_samples_split = 2)
        #roc.simple_roc(y_test, test_prob, 'ROC_RFC')
        pd.set_option('display.max_colwidth', -1)

        return render_template("rock.html", data_recall_opt=recall_opt_stats.to_html(index = False, classes = "table table-condensed"),data_precision_opt=precision_opt_stats.to_html(index = False, classes = "table table-condensed"), rocname = 'Test', recall_cm = r_cm.to_html(classes = "table table-condensed"), precision_cm = p_cm.to_html(classes = "table table-condensed"))



@app.route('/hit')
def uploaded_HTS(filename = 'MTH1_HTS_5500.csv'):
    # Read in HTS data and get features (no scores to model live app)
    df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    bits, y, ids = proc.bits_yfill_HTS(df)

    rffit = proc._unpickle('RFC_fit.pkl')
    mrecall_threshold = proc._unpickle('medianrecall_threshold.pkl')
    scored_hts = rf.score_HTS(rffit, bits, ids)
    top50 = scored_hts.iloc[:50,:]

    #Below is for validation purposes only (will only be seen for MTH1 dataset where I have to scored data)
    y_proba = rffit.predict_proba(bits)
    roc.plotHTS_roc(bits, y, y_proba, 'Test_HTS')
    precision, recall, fpr, fpr_test, tpr_test, cm = hts_a.print_threshold(rffit, bits, y, mrecall_threshold)
    cm = pd.DataFrame(cm)
    recall_opt_stats = pd.DataFrame([[format(mrecall_threshold, '.2f'),format(recall, '.2f'), format(fpr, '.2f'), format(precision, '.2f'), ]], columns = ['Suggested Threshold','True Positive Rate (Recall)', 'False Positive Rate (Fall-out)','Precision'])


    return render_template('hit.html', articles = articles, data = recall_opt_stats.to_html(index = False, classes = "table table-condensed"), hits = top50.to_html(index = False, classes = "table table-condensed"), hts_cm = cm.to_html(classes = "table table-condensed") )

    '''@app.route('/uploads/<filename>',methods=['GET', 'POST'])
    def uploaded_HTS(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'],
                                   filename)
    '''
    '''
    client = MongoClient()
    #set database that collection is in
    db = client.fragments
    #add json call to mongo
    db.fragments.insert_many(df)
    #db.review.update(data, data, upsert=True)
    '''


@app.route('/about')
def articles():
    return render_template('about.html', articles = articles)

if __name__ == '__main__':
    app.run(debug=True)
