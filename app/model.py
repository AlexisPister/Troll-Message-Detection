#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:25:30 2019

@author: alexis
"""
from sklearn.externals import joblib
import re
from nltk import PorterStemmer
from nltk.corpus import stopwords


# Use of the stems of the words, and remove the stop words
def Tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    porter_stemmer=PorterStemmer()
    words = [word for word in words if word not in stopwords.words('english')]
    words = [porter_stemmer.stem(word) for word in words]
    return words
