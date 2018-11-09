import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn import preprocessing

TEST_SIZE = 0.3


def prepare(X):
    X = X.drop(['PostCreationDate', 'OwnerUserId'], axis=1)
    if 'PostClosedDate' in X.columns:
        X = X.drop('PostClosedDate', axis=1)

    # prepare account creation dates
    def date_string_to_timestamp(date):
        try:
            return datetime.strptime(date, "%m/%d/%Y %H:%M:%S").timestamp()
        except ValueError:  # some dates are in different format
            return datetime.strptime(date, "%Y-%m-%d").timestamp()

    X["OwnerCreationDate"] = X["OwnerCreationDate"].map(date_string_to_timestamp)

    # Creating new column counting tags (more tags -> better post)
    for tag_index in ['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']:
        X[tag_index] = X[tag_index].map(lambda x: "" if isinstance(x, float) else x)

    tags = (X['Tag1'] + " " + X['Tag2'] + " " + X['Tag3'] + " " + X['Tag4'] + " " + X['Tag5']).str.strip()
    X['Tags'] = pd.Series(tags)

    X['CountTags'] = tags.map(lambda tags_str: len(tags_str.split(" ")))
    X = X.drop(['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5'], axis=1)

    return X


def get_train_data():
    Z = pd.read_csv('train.csv', skipinitialspace=True, sep=',', index_col='PostId')
    X = prepare(Z.drop('OpenStatus', axis=1))
    Y = Z['OpenStatus']
    return X, Y


def get_test_data():
    return prepare(pd.read_csv('test.csv', skipinitialspace=True, sep=',', index_col='PostId'))

# def get_normalised_numerical_data():
#     X_scaled = preprocessing.MinMaxScaler().fit_transform(X.values)
#     return train_test_split(pd.DataFrame(X_scaled), Y, test_size=TEST_SIZE)
#
# def get_numerical_data():
#     return train_test_split(X, Y, test_size=TEST_SIZE)
#
#
# def get_text_data():
#     return train_test_split(text_X, Y, test_size=TEST_SIZE)
