from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer, make_column_transformer
import testing

if __name__ == '__main__':
    clf1 = SGDClassifier(alpha=0.01, loss='modified_huber', penalty='none')
    clf2 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
    clf3 = LinearSVC()
    clf = make_pipeline(
        make_column_transformer(
            ('Title', make_pipeline(CountVectorizer(), TfidfTransformer())),
            ('BodyMarkdown', make_pipeline(CountVectorizer(), TfidfTransformer())),
            ('Tags', CountVectorizer()),
            (['ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'], MinMaxScaler()),
        ),
        VotingClassifier(estimators=[('sgd', clf1), ('lr', clf2), ('svc', clf3)], voting='hard'),
    )

    testing.cv(clf)
