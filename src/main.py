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

    model1 = models.logistic(X_train, Y_train)
    models.test_and_Evaluate(model1, X_test, Y_test)

    model2 = models.grad_boost(X_train, Y_train)
    models.test_and_Evaluate(model2, X_test, Y_test)

    model3 = models.forrest(X_train, Y_train)
    models.test_and_Evaluate(model3, X_test, Y_test)

if __name__ == "__main__":
    main()
