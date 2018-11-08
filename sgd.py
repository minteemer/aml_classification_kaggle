from sklearn.model_selection import GridSearchCV

import data_preparation
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

alpha = 0.01
loss = 'modified_huber'
penalty = 'none'

X_train, X_test, y_train, y_test = data_preparation.get_normalised_numerical_data()

def run():
    m = SGDClassifier(alpha=alpha, penalty=penalty, loss=loss)
    m.fit(X_train, y_train)
    Y_hat = m.predict(X_test)
    accuracy = accuracy_score(Y_hat, y_test)
    print('Accuracy for sgd - %.3f' % accuracy)


def cv():
    tuned_parameters = [{'loss': ['hinge', 'log', 'modified_huber'],
                         'penalty': ['none', 'l2', 'l1', 'elasticnet'],
                         'alpha': [0.01, 0.001, 0.0001],
                         }]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SGDClassifier(), tuned_parameters, cv=5,
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
