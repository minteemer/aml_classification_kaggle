from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import FunctionTransformer, MultiLabelBinarizer, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer, make_column_transformer
import pandas as pd
import testing
import data_preparation

if __name__ == '__main__':
    columns = make_column_transformer(
        ('Title', make_pipeline(CountVectorizer(), TfidfTransformer())),
        ('BodyMarkdown', make_pipeline(CountVectorizer(), TfidfTransformer())),
        ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer())),
        (['ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'], MinMaxScaler())
    )

    X, y = data_preparation.get_train_data()
    data = columns.fit_transform(X)
    testing.tune(data, y, LogisticRegression(),
                 parameters=[{
                     'penalty': ['l2'],
                     'tol': [0.0001],
                     'C': [0.1, 0.25, 0.3, 0.5, 0.7, 1.0],
                     'solver': ['liblinear'],
                     'multi_class': ['ovr']
                 }]
                 )

"""
solver : str, {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}, \
             default: 'liblinear'.

        Algorithm to use in the optimization problem.

        - For small datasets, 'liblinear' is a good choice, whereas 'sag' and
          'saga' are faster for large ones.
        - For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
          handle multinomial loss; 'liblinear' is limited to one-versus-rest
          schemes.
        - 'newton-cg', 'lbfgs' and 'sag' only handle L2 penalty, whereas
          'liblinear' and 'saga' handle L1 penalty.

        Note that 'sag' and 'saga' fast convergence is only guaranteed on
        features with approximately the same scale. You can
        preprocess the data with a scaler from sklearn.preprocessing.
 
 multi_class : str, {'ovr', 'multinomial', 'auto'}, default: 'ovr'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. For 'multinomial' the loss minimised is the multinomial loss fit
        across the entire probability distribution, *even when the data is
        binary*. 'multinomial' is unavailable when solver='liblinear'.
        'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
        and otherwise selects 'multinomial'.

        .. versionadded:: 0.18
           Stochastic Average Gradient descent solver for 'multinomial' case.
        .. versionchanged:: 0.20
            Default will change from 'ovr' to 'auto' in 0.22.
"""
