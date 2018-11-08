from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import data_preparation

criterion = 'gini'
max_depth = 8
n_estimators = 17

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf',
                      GradientBoostingClassifier(n_estimators=25, learning_rate=0.005, max_depth=7)), ])

X_train, X_test, y_train, y_test = data_preparation.get_text_data()
X_train = X_train["BodyMarkdown"]
X_test = X_test["BodyMarkdown"]
text_clf.fit(X_train, y_train)
Y_hat = text_clf.predict(X_test)
accuracy = accuracy_score(Y_hat, y_test)
print('Accuracy for SGD - %.3f' % accuracy)
