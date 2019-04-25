import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, cross_validate, cross_val_predict, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD

# Models
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

#%% Loading and processing data
data = pd.read_json(open('./data.json', 'r'), lines=True)
dataArray = data.values

X = dataArray[:,1]
Y_raw = dataArray[:,0]

def f(dico, key):
    return dico[key][0]

fvect = np.vectorize(f)

Y = fvect(Y_raw, 'label')
Y = Y.astype(int)

#%% Pipeline of tests of different classifiers

# Some classifiers
clfs = {
'CART': DecisionTreeClassifier(random_state=1),
'RF': RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=4),
'ID3': DecisionTreeClassifier(criterion = 'entropy', random_state=1),
'MLP': MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(200,100),random_state=1),
'KPPV': KNeighborsClassifier(n_neighbors=7),
'BAGGING': BaggingClassifier(n_estimators=50,random_state=1),
'ADABOOST': AdaBoostClassifier(n_estimators=50, random_state=1),
'SVC': SVC(gamma='scale', decision_function_shape='ovo'),
}

# Test several classifiers with k-fold validation
def run_classifiers(clfs,X,Y,pipeline, k=10):
    # Preprocessing
    Xproc = pipeline.fit_transform(X)
    # Cross Validation
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    scoring = ['accuracy', "roc_auc"]
    for i in clfs:
        try:
            clf = clfs[i]
            print("\n\n======= {0} =======".format(i))
            scores = cross_validate(clf, Xproc, Y, cv=kf, scoring=scoring)
            print("mean execution time : ", np.mean(scores['fit_time'] + scores['score_time']))
            print("mean accuracy : ",np.mean(scores['test_accuracy']))
            print("mean AUC : ",np.mean(scores['test_roc_auc']))
        except Exception as e:
            print(e)

#%% Preprocessing

# Use of the stems of the words, and remove the stop words
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=nltk.PorterStemmer()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [porter_stemmer.stem(word) for word in words]
    return words

# Preprocessing pipeline : Bag of Words + Tf-idf + SVD
text_proc = Pipeline([('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3), tokenizer=Tokenizer)),
                      ('svd', TruncatedSVD(algorithm='randomized', n_components=300, random_state=1))])


#%% Test
run_classifiers(clfs,X,Y,text_proc, k=4)

#%% Test the parameters of RandomForest

parameters_RF = {'n_estimators' : (2,5,10,20,30,50,100,200),
              'criterion': ("gini", "entropy"),
              'max_depth' : (None, 8)}

rf = RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=10)

X_proc = text_proc.fit_transform(X)
gs_RF = GridSearchCV(rf, parameters_RF, cv=5, scoring="roc_auc")
gs_RF.fit(X_proc,Y)

#%% We select RF as it gives the best results : we train the model on all data and save it

final_pipe = Pipeline([('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3), tokenizer=Tokenizer)),
                      ('svd', TruncatedSVD(algorithm='randomized', n_components=300, random_state=1)),
                      ('rf', RandomForestClassifier(n_estimators=200, random_state=1, n_jobs=10, criterion= 'entropy'))])

final_pipe.fit(X, Y)
joblib.dump(final_pipe, 'best_model.pkl')

#%% We save a small model which stil give good results for size reasons

small_pipe = Pipeline([('tfidf', TfidfVectorizer(min_df=.0025, max_df=0.25, ngram_range=(1,3))),
                      ('svd', TruncatedSVD(algorithm='randomized', n_components=300, random_state=1)),
                      ('rf', RandomForestClassifier(n_estimators=5, random_state=1, criterion= 'gini'))])

small_pipe.fit(X, Y)
joblib.dump(small_pipe, 'small_model.pkl')



