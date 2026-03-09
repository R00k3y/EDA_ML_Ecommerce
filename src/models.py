import pandas as pd
import numpy as np

from data_ingestion import categorical_var, numeric_var, transformed_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# logistic regression model
def logistic(x_train,y_train):
    log_pipeline = Pipeline([
        ("transformed_data", transformed_data),
        ("model",LogisticRegression(max_iter=10000))
    ])

    log_params = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__solver": ["lbfgs", "liblinear"]
    }

    log_grid = GridSearchCV(
        log_pipeline, 
        log_params,
        cv = 5, 
        scoring="accuracy",
        n_jobs=-1
    )

    fitted = log_grid.fit(x_train,y_train)
    return fitted


# gradient boosteed classification model
# function includes one-hot encoder 
def grad_boost(x_train, y_train):

    grad_pipeline = Pipeline([
        ("transformed_data", transformed_data),
        ("grad_boost_model", GradientBoostingClassifier())
    ])

    grad_boost_params = {
        "grad_boost_model__n_estimators": [100, 200],
        "grad_boost_model__learning_rate": [0.01, 0.1],
        "grad_boost_model__max_depth": [3, 5]
    }

    grad_boost_grid = GridSearchCV(
        grad_pipeline,
        grad_boost_params,
        cv = 5, 
        scoring= "accuracy",
        n_jobs=-1
    )

    fitted = grad_boost_grid.fit(x_train, y_train)

    return fitted

# random forrest model 
def forrest(x_train,y_train):

    forrest_pipeline = Pipeline([
        ("transformed_data", transformed_data),
        ("model", RandomForestClassifier(random_state=42))
    ])

    forrest_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }

    forrest_grid = GridSearchCV(
        forrest_pipeline,
        forrest_params,
        cv =5,
        scoring="accuracy",
        n_jobs=-1
    )

    fitted = forrest_grid.fit(x_train, y_train)
    return fitted
    

# function to test and evaluate a fitted model
def test_and_Evaluate(model, x_test,y_test):
    best_model = model.best_estimator_
    prediction = best_model.predict(x_test)

    print("Best Parameters:", model.best_params_)
    print("Accuracy:", best_model.score(x_test, y_test))
    print(classification_report(y_test, prediction))

# function to perform predictions with new data using a trained model 
def predict_new(model,data):
    predict = model.predict(data)
    return predict

