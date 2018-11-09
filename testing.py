from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
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


def tune(X, y, model, parameters):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, parameters, cv=5,
                           scoring='%s_macro' % score, n_jobs=-1)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()