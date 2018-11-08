from sklearn.model_selection import GridSearchCV

import data_preparation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

criterion = 'gini'
max_depth = 8
n_estimators = 17

X_train, X_test, y_train, y_test = data_preparation.get_normalised_numerical_data()

def run():
    m = RandomForestClassifier(max_depth=max_depth, criterion=criterion, n_estimators=n_estimators)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    accuracy = accuracy_score(Y_hat, y_test)
    print('Accuracy for random forest - %.3f' % accuracy)


def cv():
    tuned_parameters = [{'criterion': ['gini'],
                         'max_depth': [8],
                         'n_estimators': [16, 17, 18, 19, 20]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score, n_jobs=4)
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
    # cv()
    run()
