from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import data_preparation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

N = 19  # 17?
p = 2  # 1?

X_train, X_test, y_train, y_test = data_preparation.get_normalised_numerical_data()


def run():
    # C=100.01, kernel='rbf', degree=5, gamma=1, coef0=1
    m = SVC()
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    accuracy = accuracy_score(Y_hat, y_test)
    print('Accuracy for SVM - %.3f' % accuracy)


def cv():
    tuned_parameters = [{
        'C': [1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [1, 3, 5],
        'gamma': ['auto'],
        'coef0': [0]
    }]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score, n_jobs=8)
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


if __name__ == '__main__':
    cv()
