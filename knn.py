from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import data_preparation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = data_preparation.get_normalised_numerical_data()

m = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', KNeighborsClassifier(n_neighbors=100, p=10))
])

m.fit(X_train, y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, y_test)
print('Accuracy for KNN - %.3f' % accuracy)
