from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

import data_preparation

X_train, X_test, Y_train, Y_test = data_preparation.get_numerical_data()

m = SGDClassifier(loss="log", penalty="l2", max_iter=4, alpha=0.1)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for SGD - %.3f' % accuracy)

kernel = 'linear', 'rbf', 'poly'
m = SVC(C=100.01, kernel='rbf', degree=5, gamma=1, coef0=1)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for SVM - %.3f' % accuracy)

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
