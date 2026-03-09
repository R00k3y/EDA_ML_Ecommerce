import data_ingestion as di
import models 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    data = di.connect_db('data/online_shopping.db','online_shopping')
    data = di.clean_data(data)
    data = di.transform_data(data)

    X = data.drop('PurchaseCompleted', axis = 1)
    Y = data['PurchaseCompleted']

    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
    )

    log_regression = models.logistic(X_train, Y_train)
    models.test_and_Evaluate(log_regression, X_test, Y_test)

    grad_boost = models.grad_boost(X_train, Y_train)
    models.test_and_Evaluate(grad_boost, X_test, Y_test)

    random_forrest = models.forrest(X_train, Y_train)
    models.test_and_Evaluate(random_forrest, X_test, Y_test)

if __name__ == "__main__":
    main()
    
    # The function below is to use a trained model to predict new data
    # predict_new(trained model, new data)
