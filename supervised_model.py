# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 19:39:31 2023

@author: Vincenzo
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import warnings

algorithm_map = {
    'logistic_regression': LogisticRegression,
    'knn': KNeighborsClassifier,
    'decision_tree': DecisionTreeClassifier,
    'random_forest': RandomForestClassifier,
    'svm': SVC,
    'naive_bayes': GaussianNB,
    'perceptron': Perceptron,
    'neural_network': MLPClassifier
}

# to evaluate a single model
def train_evaluate_model(X,y, test_size, random_state,algorithm):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model_class = algorithm_map.get(algorithm)
    if model_class:
        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{algorithm} algorithm accuracy: {acc:.2f}")
    else:
        print("Invalid algorithm")
    return X_train, y_train

#to evaluate multiple models
def train_evaluate_multiple_models(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    algorithms = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), 
                  RandomForestClassifier(), SVC(), GaussianNB(), Perceptron(), MLPClassifier()]
    algorithm_names = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "Naive Bayes", "Perceptron", "Neural Network"]
   

    for model, name in zip(algorithms, algorithm_names):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro')
                recall = recall_score(y_test, y_pred, average='macro')
                f1 = f1_score(y_test, y_pred, average='macro')
                training_time = end_time - start_time
                print(f"Classifier: {name}")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1 Score: {f1:.2f}")
                print(f"Training Time: {training_time:.2f} seconds")
                print("------------------------")
            except ConvergenceWarning:
                pass
                
  
        
        
#to evaluate multiple models starting from training dataset
def train_evaluate_multiple_models_splitted(X_train, y_train,X_test, y_test, random_state):


    algorithms = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), 
                  RandomForestClassifier(), SVC(), GaussianNB(), Perceptron(), MLPClassifier()]
    algorithm_names = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM", "Naive Bayes", "Perceptron", "Neural Network"]
    results={}
    # model_results=[]
    for model, name in zip(algorithms, algorithm_names):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                results[name] = acc
                # model_results[name]= model
            except ConvergenceWarning:
                pass
                
        # Sorting the dictionary by values
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    first_key, first_value = sorted_results.popitem()
    for name, acc in sorted_results.items():
        print(f"{name} accuracy: {acc:.2f}")
    return first_key
    
    