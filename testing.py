from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

import pandas as pd
import numpy as np

import data_preparation

TEST_SIZE = 0.3

X, y = data_preparation.get_train_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

# Data used by ensemble
X_test_super_model, y_test_super_model = [], []
Y_hats_test = []

models = {
    'sgd': {
        "base_model": SGDClassifier(),
        "params": {'loss': ['hinge', 'log', 'modified_huber'],
                   'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                   'alpha': [0.01, 0.001, 0.0001],
                   }
    },

    'logistic': {
        "base_model": LogisticRegression(),
        "params": {'solver': ['liblinear', 'saga', 'lbfgs'],
                   'tol': [0.01, 0.001, 0.0001],
                   'penalty': ['l2', 'l1'],
                   'C': [0.7, 0.5, 0.3, 0.1],
                   'max_iter': [100, 200, 300]}
    },

    'svc': {
        "base_model": LinearSVC(),
        "params": {'C': [0.1, 0.2, 0.3, 0.4, 0.5]}
    },
}

optimal_parameters = {
    'sgd': {'loss': ['hinge'],
            'penalty': ['l2'],
            'alpha': [0.0001],
            },
    'logistic': {'solver': ['lbfgs'],
                 'tol': [0.0001],
                 'penalty': ['l2'],
                 'C': [0.3],
                 'max_iter': [300]},
    'svc': {'C': [0.1]}
}


def prepare_ensemble_data():
    """
    This method divides train data into 4 parts:
    50% - X_train, y_train
    25% - X_test, y_test for checking models fitted with X_train and y_train
    25% - X_test_super_model, y_test_super_model - testing predictions of different models with super model predictions
    """
    global X_train, X_test, y_train, y_test, X_test_super_model, y_test_super_model

    X_train, X_test_half, y_train, y_test_half = train_test_split(X, y, test_size=0.5)
    X_test, X_test_super_model, \
    y_test, y_test_super_model = train_test_split(X_test_half, y_test_half, test_size=0.5)


def ensemble_test_estimator(model):
    """
    Performs ensemble testing of the base estimator model (fit and predict),
    prints accuracy results and saves them in Y_hats_test matrix.
    :param model: given super model
    :return: output predicted by base estimator model
    """
    global Y_hats_test, X_test_super_model

    model.fit(X_train, y_train)
    Y_hat = model.predict(X_test)
    accuracy = accuracy_score(Y_hat, y_test)
    print('Accuracy: %.3f' % accuracy)
    Y_hats_test.append(single_test(model, X_test_super_model))
    return Y_hat


def single_test(model, x_test):
    """
    Uses model to predict output
    :param model: given model
    :param x_test: input data on which prediction will be based
    :return: predicted output
    """
    Y_hat = model.predict(x_test)
    return Y_hat


def cv_score(model):
    """
    Performs cross validation on the chosen model and prints mean accuracy for each iteration.
    """
    cros_val_sores = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    print("Average score: %.3f" % np.mean(cros_val_sores))
    print(cros_val_sores)


def generate_solution(model):
    """
    Uses given model to predict outputs and save them into 'solution.csv'.
    :param model: chosen model
    """
    X_test = data_preparation.get_test_data()
    model.fit(X, y)
    Y_hat = model.predict(X_test)
    solution = pd.DataFrame(Y_hat, columns=['OpenStatus'], index=X_test.index)
    solution.to_csv('solution.csv')


def grid_search(model, tuned_params_dict, print_results=True):
    """
    Performs grid search cross validation to find optimal parameters for the given model.
    Prints cross validation results.
    :param model: given model
    :param tuned_params_dict: dictionary of parameters to check
    :param print_results: if True - prints results to console (default: True)
    """
    global X, y

    # Use Grid Search
    tuned_parameters = [tuned_params_dict]
    clf = GridSearchCV(model, tuned_parameters, cv=5, scoring='accuracy', n_jobs=-1)
    clf.fit(X, y)

    # Assign results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    # Print results if needed
    if print_results:
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


def ensemble_test_super_model(Y_hats, model):
    """
    Performs ensemble method on the given y's that were taken from base estimator models.
    Prints results of ensemble.
    :param Y_hats: outputs taken from base estimator models.
    :param model: chosen super model
    """
    global y_test_super_model, y_test, Y_hats_test

    # Y_hats = X_test_super
    # One hot encode Y_hats
    Y_hats = pd.get_dummies(Y_hats)

    model.fit(Y_hats, y_test)

    # Build and one hot encode testing data
    Y_hats_test = {'Y_hat1': Y_hats_test[0], 'Y_hat2': Y_hats_test[1], 'Y_hat3': Y_hats_test[2]}
    Y_hats_test = pd.DataFrame(Y_hats_test)
    Y_hats_test = pd.get_dummies(Y_hats_test)

    Y_hat = model.predict(Y_hats_test)
    accuracy = accuracy_score(Y_hat, y_test_super_model)
    print('Accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    # ---- Uncomment needed tool ----
    grid_search(models['logistic']['base_model'], models['logistic']['params'])