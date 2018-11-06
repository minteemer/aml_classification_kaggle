import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from common.visualize.method import plot_predictions_2D, show_param_dependency

# Make Y and X train and test sets
train = pd.read_csv('train.csv')
Y = train['OpenStatus']
# X_test = pd.read_csv('test.csv')
X = train.drop('OpenStatus', axis=1)

# TODO: analyze dates
# But for now simply drop them
X = X.drop('PostClosedDate', axis=1).drop('PostCreationDate', axis=1) \
    .drop('PostId', axis=1).drop('OwnerCreationDate', axis=1)

# TODO: analyze titles and bodyMarkdowns considering tags and not only
# But for now simply drop them
X = X.drop('Title', axis=1).drop('BodyMarkdown', axis=1)

# Creating new column counting tags (more tags -> better post)
X['Tag1'] = X['Tag1'].map(lambda x: 0 if isinstance(x, float) else 1)
X['Tag2'] = X['Tag2'].map(lambda x: 0 if isinstance(x, float) else 1)
X['Tag3'] = X['Tag3'].map(lambda x: 0 if isinstance(x, float) else 1)
X['Tag4'] = X['Tag4'].map(lambda x: 0 if isinstance(x, float) else 1)
X['Tag5'] = X['Tag5'].map(lambda x: 0 if isinstance(x, float) else 1)
count_tags = []
for index, row in X.iterrows():
    count_tags.append(row['Tag1'] + row['Tag2'] + row['Tag3'] + row['Tag4'] + row['Tag5'])
X['CountTags'] = count_tags
X = X.drop('Tag1', axis=1).drop('Tag2', axis=1).drop('Tag3', axis=1) \
    .drop('Tag4', axis=1).drop('Tag5', axis=1)

# TODO cross validation
# For now simple splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# Building different models
m = KNeighborsClassifier(n_neighbors=100, p=10)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for KNN - %.3f' % accuracy)

m = SGDClassifier(loss="log", penalty="l2", max_iter=4, alpha=0.1)
m.fit(X_train, Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat, Y_test)
print('Accuracy for SGD - %.3f' % accuracy)

# kernel='linear', 'rbf', 'poly'
# m = SVC(C=100.01, kernel='rbf', degree=5, gamma=1, coef0=1)
# m.fit(X_train, Y_train)
# Y_hat = m.predict(X_test)
# accuracy = accuracy_score(Y_hat, Y_test)
# print('Accuracy for SVM - %.3f' % accuracy)

m = DecisionTreeClassifier(max_depth=4)
m.fit(X_train,Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat,Y_test)
print('Accuracy for decision tree - %.3f' % accuracy)

# n_estimators: 1,3,100
# max_depth=None,1,2,5,10
m = RandomForestClassifier(max_features=0.5, n_estimators=100, bootstrap=True,max_depth=5)
m.fit(X_train,Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat,Y_test)
print('Accuracy for random forest regressor - %.3f' % accuracy)

# max_depth:1,2,
# n_estimators:1,3,2000
m = GradientBoostingClassifier(n_estimators=10, learning_rate=0.2,  max_depth=4)
m.fit(X_train,Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat,Y_test)
print('Accuracy for gradient boosting - %.3f' % accuracy)

# solver: sgd, lbfgs, adam
# hidden_layer_sizes: [4],[40],[400],[50,50,50]
# activation: identity, logistic, tanh, relu
m = MLPClassifier(hidden_layer_sizes=[40], alpha=0.00100, max_iter=1000, activation='relu', solver='lbfgs')
m.fit(X_train,Y_train)
Y_hat = m.predict(X_test)
accuracy = accuracy_score(Y_hat,Y_test)
print('Accuracy for MLP - %.3f' % accuracy)