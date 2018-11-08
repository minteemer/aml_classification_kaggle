from sklearn.model_selection import GridSearchCV

import data_preparation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

N = 19  # 17?
p = 2  # 1?

X_train, X_test, y_train, y_test = data_preparation.get_normalised_numerical_data()


def run():
    m = KNeighborsClassifier(n_neighbors=N, p=p, n_jobs=8)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    accuracy = accuracy_score(Y_hat, y_test)
    print('Accuracy for KNN - %.3f' % accuracy)


#4 0.297 (+/-0.036) for {'n_neighbors': 19, 'p': 2}
def cv():
    tuned_parameters = [{
        'n_neighbors': [17, 19],
        'p': [1, 2]
    }]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score, n_jobs=8)
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
    run()
