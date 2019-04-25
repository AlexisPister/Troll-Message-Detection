#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:25:30 2019

@author: alexis
"""
from sklearn.externals import joblib

clf = joblib.load(open("static/small_model.pkl", "rb"))
