# Troll-d

Troll-d is an application hosted on the google cloud app engine, which predicts if a message can be considered as a  "troll" message, with a Random Forest algorithm.

## The model

The raw text go through a Bag of Words and a Tf-Idf transformation. A truncated SVD is then applied to reduce the dimensionnality and get a vector of 300 dimensions describing the text. This vector is then given to a trained and optimized Random Forest Algorithm, labeling the message as "troll" or "not troll". An *AUC of 0.965* is achieved on the validation set.

The notebook modelStudy.ipynb is a study to find the algorithm and parameters giving the best AUC possible (it should be opened with spyder to have control over the cells).

## The app

The model is hosted on a Flask application on the google cloud app engine. The app folder contains all the files hosted on the cloud.

* main.py : flask application
* app.yaml : configuration file
* static : contains the machine learning pipeline and the .css
* templates : html files
