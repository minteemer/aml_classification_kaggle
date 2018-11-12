import numpy

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.compose import make_column_transformer

import pandas as pd

import data_preparation
import testing


def ensemble_models():
    testing.prepare_ensemble_data()

    clf1 = make_pipeline(
        make_column_transformer(
            ('Title', make_pipeline(CountVectorizer(min_df=0.00007), TfidfTransformer())),
            ('BodyMarkdown', make_pipeline(CountVectorizer(max_df=0.8, min_df=0.00007), TfidfTransformer())),
            ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer())),
            (['ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'], MinMaxScaler()),
        ),
        SGDClassifier(loss='log', alpha=0.0001, penalty='l2', power_t=0.15),
    )
    Y_hat1 = testing.ensemble_test_estimator(clf1)
    print('--- SGD DONE ---')

    clf2 = make_pipeline(
        make_column_transformer(
            ('Title', make_pipeline(CountVectorizer(min_df=0.00007), TfidfTransformer())),
            ('BodyMarkdown', make_pipeline(CountVectorizer(min_df=0.00007), TfidfTransformer())),
            ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer())),
            (['ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'], MinMaxScaler()),
        ),
        LogisticRegression(penalty='l2', tol=0.0001, solver='liblinear', multi_class='ovr', C=0.3),
    )
    Y_hat2 = testing.ensemble_test_estimator(clf2)
    print('--- Logistic DONE ---')

    clf3 = make_pipeline(
        make_column_transformer(
            ('Title', make_pipeline(CountVectorizer(min_df=0.00007), TfidfTransformer())),
            ('BodyMarkdown', make_pipeline(CountVectorizer(min_df=0.00007), TfidfTransformer())),
            ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer())),
            (['ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'], MinMaxScaler()),
        ),
        LinearSVC(C=0.1),
    )
    Y_hat3 = testing.ensemble_test_estimator(clf3)
    print('--- SVC DONE ---')

    Y_hat_matrix = pd.DataFrame(data={'Y_hat1': Y_hat1, 'Y_hat2': Y_hat2, 'Y_hat3': Y_hat3})
    testing.ensemble_test_super_model(Y_hat_matrix, SGDClassifier())
    print('--- Super model SGD DONE ---')


def voting():
    clf1 = SGDClassifier(loss='hinge', alpha=0.0001, penalty='l2', power_t=0.15, n_jobs=-1)
    clf2 = LogisticRegression(penalty='l2', tol=0.0001, solver='lbfgs', C=0.3, max_iter=300, n_jobs=-1)
    clf3 = LinearSVC(C=0.1)

    eclf = make_pipeline(
        make_column_transformer(
            ('Title', make_pipeline(CountVectorizer(), TfidfTransformer(norm="l2"))),
            ('BodyMarkdown', make_pipeline(CountVectorizer(), TfidfTransformer(norm="l2"))),
            ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer(norm="l2"))),
            (
            ['ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'], StandardScaler()),
        ),
        VotingClassifier(estimators=[('sgd', clf1), ('lr', clf2), ('svc', clf3)], n_jobs=-1),
    )

    testing.cv_score(eclf)
    testing.generate_solution(eclf)


def ada_boost():
    sgd = SGDClassifier(loss='hinge', alpha=0.0001, penalty='l2', power_t=0.15, n_jobs=-1)
    logistics = LogisticRegression(penalty='l2', tol=0.0001, solver='lbfgs', multi_class='ovr', C=0.3, max_iter=300,
                              n_jobs=-1)
    svc = LinearSVC(C=0.1)

    eclf = make_pipeline(
        make_column_transformer(
            ('Title', make_pipeline(CountVectorizer(), TfidfTransformer())),
            ('BodyMarkdown', make_pipeline(CountVectorizer(), TfidfTransformer())),
            ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer())),
            (
                ['CountTags', 'ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'],
                StandardScaler()
            ),
        ),
        AdaBoostClassifier(base_estimator=sgd, n_estimators=100, learning_rate=0.5, algorithm='SAMME'),
    )
    testing.cv_score(eclf)


if __name__ == '__main__':
    # ensemble_models()
    # ada_boost()
    voting()
