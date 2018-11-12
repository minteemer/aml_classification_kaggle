import pandas as pd
from datetime import datetime


def prepare(X):
    # Drop useless parameters
    X = X.drop(['PostCreationDate', 'OwnerUserId'], axis=1)
    if 'PostClosedDate' in X.columns:
        X = X.drop('PostClosedDate', axis=1)

    # Convert account creation dates to numerical format (timestamp)
    def date_string_to_timestamp(date):
        try:
            return datetime.strptime(date, "%m/%d/%Y %H:%M:%S").timestamp()
        except ValueError:  # a few dates are in different format
            return datetime.strptime(date, "%Y-%m-%d").timestamp()

    X["OwnerCreationDate"] = X["OwnerCreationDate"].map(date_string_to_timestamp)

    # Replace NaN in TagN columns to empty string
    tags_columns = ['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']
    for tag_index in tags_columns:
        X[tag_index] = X[tag_index].map(lambda x: "" if isinstance(x, float) else x)

    # Create column that contains all tags as one string
    tags = (X['Tag1'] + " " + X['Tag2'] + " " + X['Tag3'] + " " + X['Tag4'] + " " + X['Tag5']).str.strip()
    X['Tags'] = pd.Series(tags)

    # Create column that contains number of tags
    X['CountTags'] = tags.map(lambda tags_str: len(tags_str.split(" ")))
    X = X.drop(tags_columns, axis=1)

    return X


def get_train_data():
    Z = pd.read_csv(r'data/train.csv', skipinitialspace=True, sep=',', index_col='PostId')
    X = prepare(Z.drop('OpenStatus', axis=1))
    Y = Z['OpenStatus']
    return X, Y


def get_test_data():
    return prepare(pd.read_csv(r'data/test.csv', skipinitialspace=True, sep=',', index_col='PostId'))
