from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import numpy as np
import data_preparation

TEST_SIZE = 0.3


def single_test(model):
    X, y = data_preparation.get_train_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    model.fit(X_train, y_train)
    Y_hat = model.predict(X_test)
    accuracy = accuracy_score(Y_hat, y_test)
    print('Accuracy: %.3f' % accuracy)


def cv(model):
    X_train, y_train = data_preparation.get_train_data()
    cros_val_sores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


def generate_solution(model):
    X_test = data_preparation.get_test_data()
    Y_hat = model.predict(X_test)
    S = pd.DataFrame(Y_hat, columns=['OpenStatus'], index=X_test.index)
    print(S.head())
    S.to_csv('solution.csv')  # accuracy>0.49 on test
