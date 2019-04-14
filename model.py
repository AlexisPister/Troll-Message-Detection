import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.externals import joblib

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

#%% Splitting data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words="english",
                     min_df=.0025, max_df=0.25, ngram_range=(1,3))),
                      ('clf', RandomForestClassifier())])

# Training
text_clf.fit(X_train, y_train)
# Test
preds = text_clf.predict(X_test)

#%% Metrics
print("Accuracy:", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall:", recall_score(y_test, preds))
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

#%% Export
joblib.dump(text_clf, 'troll_model.pkl')


