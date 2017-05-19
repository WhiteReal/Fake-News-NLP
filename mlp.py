#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 17:33:18 2017

@author: white
"""

from utils.dataset import DataSet
from utils.generate_test_splits import split
from utils.score import report_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.externals import joblib

def stop_list(filepath):
    stoplist = []
    with open(filepath, "r") as stop_file:
        for line in stop_file:
            stoplist.append(line.strip())
    return stoplist


dataset = DataSet()
data_splits = split(dataset)

training_data = data_splits['training']
dev_data = data_splits['dev']
test_data = data_splits['test']

related = []
label = []
for stance in training_data:
    if stance["Stance"] != "unrelated":
        related.append(stance["Headline"] + " " + dataset.articles[stance["Body ID"]])
        label.append(stance["Stance"])
        
stoplist = stop_list("stop_list.txt")
count_vect = CountVectorizer(stop_words=stoplist)
vec_result = count_vect.fit_transform(related).toarray()
        
clf = MLPClassifier(solver='lbfgs')
#clf = MLPClassifier(solver='adam')
clf.fit(vec_result, label) 
joblib.dump(clf, "model_solver_lbfgs.m")

