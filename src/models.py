import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, classification_report

# logistic regression model
def logistic(x_train,y_train):
    model = LogisticRegression
    fitted = model.fit(x_train,y_train)
    return fitted


# gradient boosteed classification model
# function includes one-hot encoder 
def grad_boost(x_train, y_train):

    categorical_var = ['SpecialDayProximity', 'GeographicRegion', 'TrafficSource', 'CustomerType']
    numeric_var = ['ProductPageTime', 'BounceRate', 'ExitRate', 'PageValue']

    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_var),   # one-hot encoder 
        ("numerical", "passthrough", numeric_var)
    ])

    model = GradientBoostingClassifier()

    pipeline = Pipeline([
        ("data_preprocess", preprocess),
        ("grad_boost_model", model)
    ])

    pipeline.fit(x_train, y_train)

    return pipeline

# random forrest model 
def forrest(x_train,y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model.fit(x_train, y_train)
    

# function to test and evaluate a fitted model
def test_and_Evaluate(model, x_test,y_test):
    pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    print('Accuracy score:',accuracy)
    print(classification_report(y_test, pred))

def predict_new(model,data):
    predict = model.predict(data)
    return predict

