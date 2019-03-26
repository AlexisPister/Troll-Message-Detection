from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load(open("troll_model.pkl", "rb"))

    if request.method == "POST":
        message = request.form['message']
        data = [message]
        pred = clf.predict(data)
    return render_template("result.html", prediction=pred)


if __name__ == '__main__':
    app.run(debug= True)
