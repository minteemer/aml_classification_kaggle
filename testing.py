from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
import numpy as np
import data_preparation

TEST_SIZE = 0.3

X, y = data_preparation.get_train_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
X_test_super_model, y_test_super_model = [], []
Y_hats_test = []


def prepare_ensemble_data():
    global X_train, X_test, y_train, y_test, X_test_super_model, y_test_super_model
    X_train, X_test_half, y_train, y_test_half = train_test_split(X, y, test_size=0.5)
    X_test, X_test_super_model, \
    y_test, y_test_super_model = train_test_split(X_test_half, y_test_half, test_size=0.5)


def single_test(model):
    global Y_hats_test, X_test_super_model
    model.fit(X_train, y_train)
    Y_hat = model.predict(X_test)
    accuracy = accuracy_score(Y_hat, y_test)
    print('Accuracy: %.3f' % accuracy)
    Y_hats_test.append(sigle_certain_test(model, X_test_super_model))
    return Y_hat


def sigle_certain_test(model, x_test):
    Y_hat = model.predict(x_test)
    return Y_hat


def cv(model):
    cros_val_sores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


def generate_solution(model):
    X_test = data_preparation.get_test_data()
    model.fit(X, y)
    Y_hat = model.predict(X_test)
    S = pd.DataFrame(Y_hat, columns=['OpenStatus'], index=X_test.index)
    print(S.head())
    S.to_csv('solution.csv')


def test_y_hat(Y_hats, model):
    global y_test_super_model, y_test, Y_hats_test
    # Y_hats = X_test_super
    Y_hats = pd.get_dummies(Y_hats)
    model.fit(Y_hats, y_test)

    Y_hats_test = {'Y_hat1': Y_hats_test[0], 'Y_hat2': Y_hats_test[1], 'Y_hat3': Y_hats_test[2]}
    Y_hats_test = pd.DataFrame(Y_hats_test)
    Y_hats_test = pd.get_dummies(Y_hats_test)

    Y_hat = model.predict(Y_hats_test)
    accuracy = accuracy_score(Y_hat, y_test_super_model)
    print('Accuracy: %.3f' % accuracy)

# ---train half (X_train, y_train)--- -test 1\4 (X_test, y_test)- -test super 1\4-
