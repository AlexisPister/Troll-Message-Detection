from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib

if not("clf" in globals()) and not("clf" in locals()):
    clf = joblib.load(open("troll_model.pkl", "rb"))

application = Flask(__name__)

@application.route("/")
def home():
    return render_template('index.html')

@application.route('/predict', methods=['POST'])
def predict():
    #if not("clf" in globals()) and not("clf" in locals()):
        #clf = joblib.load(open("troll_model.pkl", "rb"))

    if request.method == "POST":
        message = request.form['message']
        data = [message]
        pred = clf.predict(data)

    return render_template("result.html", prediction=pred)


if __name__ == '__main__':
    #clf = joblib.load(open("troll_model.pkl", "rb"))
    #print("model loaded")
    application.run()
    
