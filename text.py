from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.compose import ColumnTransformer, make_column_transformer
import testing

if __name__ == '__main__':

    clf = make_pipeline(
        make_column_transformer(
            ('Title', make_pipeline(CountVectorizer(), TfidfTransformer())),
            ('BodyMarkdown', make_pipeline(CountVectorizer(), TfidfTransformer())),
            ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer())),
            (['ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'], MinMaxScaler()),
    #         (['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5'], make_pipeline(
    #             DropNaTransformer(),
    #             OneHotEncoder(),
    #         )),
        ),
        LogisticRegression(),
    )

    testing.cv(clf)

    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    # clf.fit(X_train, y_train)

# criterion = 'gini'
# max_depth = 8
# n_estimators = 17
#
# text_clf = Pipeline([('vect', CountVectorizer()),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf',
#                       GradientBoostingClassifier(n_estimators=25, learning_rate=0.005, max_depth=7)), ])
#
# text_clf.fit(X_train, y_train)
# Y_hat = text_clf.predict(X_test)
# accuracy = accuracy_score(Y_hat, y_test)
# print('Accuracy for SGD - %.3f' % accuracy)
