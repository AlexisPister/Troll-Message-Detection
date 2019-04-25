from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
import re

from nltk import PorterStemmer
from nltk.corpus import stopwords

# =============================================================================
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# =============================================================================


clf = joblib.load(open("static/best_model.pkl", "rb"))
application = Flask(__name__)


# Use of the stems of the words, and remove the stop words
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=PorterStemmer()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [porter_stemmer.stem(word) for word in words]
    return words


def give_result(message):
    pred = clf.predict(message)
    return pred

@application.route("/")
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    #if not("clf" in globals()):
    #clf = joblib.load(open("static/small_model.pkl", "rb"))
    #global clf

    if request.method == "POST":
        message = request.form['message']
        data = [message]
        print(data)
        #pred = clf.predict(data)

    return render_template("result.html", prediction=give_result(data))



if __name__ == '__main__':
    #clf = joblib.load(open("troll_model.pkl", "rb"))
    #print("model loaded")
    application.run(debug=True)

