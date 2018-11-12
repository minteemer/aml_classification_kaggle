from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import data_preparation

X, Y = data_preparation.get_train_data()
data_transformer = make_column_transformer(
    ('Title', make_pipeline(CountVectorizer(), TfidfTransformer())),
    ('BodyMarkdown', make_pipeline(CountVectorizer(), TfidfTransformer())),
    ('Tags', make_pipeline(CountVectorizer(), TfidfTransformer())),
    (
        ['CountTags', 'ReputationAtPostCreation', 'OwnerUndeletedAnswerCountAtPostTime', 'OwnerCreationDate'],
        StandardScaler()
    ),
)
X = data_transformer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

m = SGDClassifier(loss='hinge', n_jobs=-1)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for SGD - %.3f' % accuracy)

m = LogisticRegression(n_jobs=-1)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for LogisticsRegression - %.3f' % accuracy)

m = LinearSVC(C=0.5)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for LinearSVC - %.3f' % accuracy)


m = KNeighborsClassifier(n_neighbors=17, p=2, n_jobs=8)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for KNN - %.3f' % accuracy)

m = DecisionTreeClassifier(max_depth=4)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for decision tree - %.3f' % accuracy)

# n_estimators: 1,3,100
# max_depth=None,1,2,5,10
m = RandomForestClassifier(max_features=0.5, n_estimators=100, bootstrap=True, max_depth=5)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for random forest regressor - %.3f' % accuracy)

# max_depth:1,2,
# n_estimators:1,3,2000
m = GradientBoostingClassifier(n_estimators=10, learning_rate=0.2, max_depth=4)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for gradient boosting - %.3f' % accuracy)

# solver: sgd, lbfgs, adam
# hidden_layer_sizes: [4],[40],[400],[50,50,50]
# activation: identity, logistic, tanh, relu
m = MLPClassifier(hidden_layer_sizes=[40], alpha=0.00100, max_iter=1000, activation='relu', solver='lbfgs')
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for MLP - %.3f' % accuracy)

kernel = 'linear', 'rbf', 'poly'
m = SVC(C=100.01, kernel='rbf', degree=5, gamma=1, coef0=1)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for SVM - %.3f' % accuracy)
